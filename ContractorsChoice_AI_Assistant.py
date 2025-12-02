import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
import tempfile
import re
import requests
from datetime import date
from fpdf import FPDF

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# ============================================================
# SECRETS & GLOBAL CONFIG
# ============================================================

# Populate env vars from Streamlit secrets when running on Streamlit Cloud
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# Optional: Twilio for SMS
TWILIO_ACCOUNT_SID = st.secrets.get("TWILIO_ACCOUNT_SID", None)
TWILIO_AUTH_TOKEN = st.secrets.get("TWILIO_AUTH_TOKEN", None)
TWILIO_FROM_NUMBER = st.secrets.get("TWILIO_FROM_NUMBER", None)

st.set_page_config(
    page_title="ContractorsChoice AI Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Choose LLM provider
MODEL_NAME = "gpt-4.1-mini"
LLM_PROVIDER = "OpenAI"

if os.environ.get("GROQ_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
    LLM_PROVIDER = "Groq"
    MODEL_NAME = "llama3-70b-8192"
elif not os.environ.get("OPENAI_API_KEY") and not os.environ.get("GROQ_API_KEY"):
    st.error("FATAL: Neither OPENAI_API_KEY nor GROQ_API_KEY are set. Please configure at least one.")
    st.stop()


# ============================================================
# LLM & EMBEDDINGS
# ============================================================

@st.cache_resource
def load_llm():
    if LLM_PROVIDER == "OpenAI":
        return ChatOpenAI(temperature=0, model=MODEL_NAME)
    else:
        return ChatGroq(temperature=0, model_name=MODEL_NAME)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = load_llm()
embeddings = load_embeddings()


# ============================================================
# HELPER FUNCTIONS (shared + contractor)
# ============================================================

def money(x: float) -> str:
    return f"${x:,.2f}"

def generate_proposal_text(
    contractor_name: str,
    client_name: str,
    project_name: str,
    project_location: str,
    project_description: str,
    start_date,
    end_date,
    labor_hours: float,
    labor_rate: float,
    materials_cost: float,
    overhead_cost: float,
    profit_amount: float,
    total_price: float,
    payment_terms: str,
    notes: str,
) -> str:
    """Base non-LLM proposal template."""
    start = start_date.strftime("%B %d, %Y") if start_date else "TBD"
    end = end_date.strftime("%B %d, %Y") if end_date else "TBD"

    text = f"""
{contractor_name}
Proposal for: {client_name}
Project: {project_name}
Location: {project_location}

1. Project Overview
{project_description.strip() or "Scope of work as discussed with client."}

Proposed schedule:
- Start date: {start}
- Substantial completion: {end}

2. Pricing Summary
- Estimated labor: {labor_hours:.1f} hours @ {money(labor_rate)} per hour
- Labor subtotal: {money(labor_hours * labor_rate)}
- Materials: {money(materials_cost)}
- Overhead & indirects: {money(overhead_cost)}
- Profit: {money(profit_amount)}

Total lump-sum price (before any applicable sales tax): {money(total_price)}

3. Exclusions & Assumptions
- Hidden conditions, structural defects, or code issues not visible at time of estimate are excluded.
- Permit fees, engineering, and third-party inspections are excluded unless otherwise stated in writing.
- Work will be performed during normal business hours unless otherwise agreed.

4. Payment Terms
{payment_terms.strip()}

5. Acceptance
To authorize this work, please sign below or provide written approval referencing this proposal.

Client: {client_name}
Date: ________________________

Contractor: {contractor_name}
Date: ________________________

Additional notes:
{notes.strip() or "None."}
"""
    return text.strip()

def smart_rewrite_proposal(base_text: str, tone: str = "professional") -> str:
    """Use LLM to rewrite proposal in a more polished, client-friendly tone."""
    try:
        prompt = f"""
You are an expert construction sales writer. Rewrite the following project proposal 
in a clear, persuasive, and {tone} tone, suitable to send to a homeowner or commercial client.
Preserve all pricing, dates, and scope details, but improve the wording and flow.

Proposal text:
\"\"\"{base_text}\"\"\"
"""
        resp = llm.invoke(prompt)
        return resp.content.strip()
    except Exception as e:
        return f"[SMART WRITER ERROR] {e}\\n\\nOriginal proposal:\\n\\n{base_text}"

def create_proposal_pdf(proposal_text: str) -> bytes:
    """Make a simple PDF from text and return bytes."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    for line in proposal_text.split("\\n"):
        pdf.multi_cell(0, 5, line)
    return pdf.output(dest="S").encode("latin-1")

def send_sms_via_twilio(to_number: str, body: str) -> str:
    """Send SMS using Twilio, if configured."""
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER):
        return (
            "Twilio not configured. Please set "
            "TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_FROM_NUMBER in secrets."
        )

    url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
    data = {"To": to_number, "From": TWILIO_FROM_NUMBER, "Body": body}
    try:
        resp = requests.post(url, data=data, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
        if resp.status_code in (200, 201):
            return "SMS sent successfully."
        return f"SMS failed ({resp.status_code}): {resp.text}"
    except Exception as e:
        return f"SMS error: {e}"


# ============================================================
# STATE INITIALIZATION
# ============================================================

if "dfs" not in st.session_state:
    st.session_state.dfs = {}
if "pdf_names" not in st.session_state:
    st.session_state.pdf_names = set()
if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "master_agent_executor" not in st.session_state:
    st.session_state.master_agent_executor = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Contractor state
if "contractor_tasks" not in st.session_state:
    st.session_state.contractor_tasks = []
if "contractor_notes" not in st.session_state:
    st.session_state.contractor_notes = ""
if "contractor_files" not in st.session_state:
    st.session_state.contractor_files = []
if "current_proposal" not in st.session_state:
    st.session_state.current_proposal = ""


# ============================================================
# BI DATA PROCESSING HELPERS
# ============================================================

def process_structured_data(uploaded_file):
    filename = uploaded_file.name
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            return

        df.columns = (
            df.columns
            .str.strip()
            .str.replace(r"[\\s\\-]+", "_", regex=True)
        )
        df.columns = df.columns.str.replace("__", "_", regex=False)

        rename_map = {
            "date": "Date",
            "Date": "Date",
            "product": "Product",
            "Product": "Product",
            "region": "Region",
            "Region": "Region",
            "region_sales": "Region",
            "Region_Sales": "Region",
            "sales_revenue": "Sales_Revenue",
            "sale_revenue": "Sales_Revenue",
            "Sale_Revenue": "Sales_Revenue",
            "Sale__Revenue": "Sales_Revenue",
            "Sales_Revenue": "Sales_Revenue",
        }
        df.rename(columns=rename_map, inplace=True)

        st.session_state.dfs[filename] = df
        st.success(f"Structured Data loaded: {filename}")
    except Exception as e:
        st.error(f"Error processing structured data {filename}: {e}")

def process_pdf_data(uploaded_file):
    filename = uploaded_file.name
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            path = tmp.name
        loader = PyPDFLoader(path)
        docs = loader.load()
        os.unlink(path)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        st.session_state.pdf_chunks.extend(chunks)
        st.session_state.pdf_names.add(filename)
        st.success(f"PDF loaded: {filename} ({len(chunks)} chunks).")
    except Exception as e:
        st.error(f"Error processing PDF {filename}: {e}")

def create_retriever_from_chunks():
    if st.session_state.pdf_chunks:
        vs = FAISS.from_documents(st.session_state.pdf_chunks, embeddings)
        st.session_state.retriever = vs.as_retriever()
        st.info(
            f"Unified PDF RAG index rebuilt with {len(st.session_state.pdf_chunks)} "
            f"chunks from {len(st.session_state.pdf_names)} document(s)."
        )
    else:
        st.session_state.retriever = None

def handle_file_upload(uploaded_file):
    if uploaded_file.name.endswith((".csv", ".xls", ".xlsx")):
        process_structured_data(uploaded_file)
    elif uploaded_file.name.endswith(".pdf"):
        process_pdf_data(uploaded_file)
        create_retriever_from_chunks()
    else:
        st.error("Unsupported file type. Please upload CSV, Excel, or PDF.")

    create_master_agent()
    st.session_state.chat_history = []


# ============================================================
# BI AGENT HELPERS
# ============================================================

def create_pandas_executor_func(agent_executor, filename):
    def fn(query: str) -> str:
        result = agent_executor.invoke({"input": query})
        return f"ANSWER FROM **{filename}**: " + result.get("output", "No structured answer.")
    return fn

def pdf_qa_executor_func(query: str) -> str:
    if st.session_state.retriever is None:
        return "No PDF document is currently loaded for RAG analysis."
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=st.session_state.retriever,
    )
    resp = qa_chain.invoke({"query": query})
    src = f"ANSWER FROM **UNSTRUCTURED PDF DATA** (Docs: {', '.join(st.session_state.pdf_names)})"
    return src + ": " + resp.get("result", "No answer from PDF content.")

def create_master_agent():
    tools = []

    # Structured tools
    for filename, df in st.session_state.dfs.items():
        agent_exec = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=False,
            allow_dangerous_code=True,
        )
        tool_func = create_pandas_executor_func(agent_exec, filename)
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", filename.split(".")[0])
        tools.append(
            Tool(
                name=f"structured_data_on_{sanitized}",
                description=(
                    "Use this to answer questions about structured data in "
                    f"{filename}. Input must be a full business question."
                ),
                func=tool_func,
            )
        )

    # PDF RAG tool
    if st.session_state.retriever is not None:
        tools.append(
            Tool(
                name="unstructured_pdf_document_qa",
                description=(
                    "Use this to answer questions from ALL uploaded PDF documents "
                    "(unstructured text). Input must be a full question."
                ),
                func=pdf_qa_executor_func,
            )
        )

    if not tools:
        st.session_state.master_agent_executor = None
        return

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an intelligent assistant that can use multiple tools to analyze "
                "structured CSV/Excel data and unstructured PDF documents. Always reference "
                "the source filename in your final answers."
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    master_agent = create_tool_calling_agent(llm, tools, prompt)
    st.session_state.master_agent_executor = AgentExecutor(
        agent=master_agent,
        tools=tools,
        verbose=False,
        max_iterations=15,
        handle_parsing_errors=True,
    )
    st.success(f"Master BI Agent initialized with {len(tools)} tool(s).")


# ============================================================
# BI VISUALIZATIONS
# ============================================================

def generate_visual_insights(df: pd.DataFrame):
    st.subheader("Visual Data Insights")

    if not isinstance(df, pd.DataFrame):
        st.warning("No structured data available for visualization.")
        return

    def safe_plot(title, creator, warn_msg):
        try:
            fig, ax = creator()
            ax.set_title(title, fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"{warn_msg} Error: {e}")

    def plot_sales_trend():
        if "Date" in df.columns and "Sales_Revenue" in df.columns:
            dfp = df.copy()
            dfp["Date"] = pd.to_datetime(dfp["Date"], errors="coerce")
            dfp["Sales_Revenue"] = pd.to_numeric(dfp["Sales_Revenue"], errors="coerce")
            dfp = dfp.dropna(subset=["Date", "Sales_Revenue"])
            if dfp.empty:
                raise ValueError("No valid Date/Sales_Revenue data.")
            monthly = dfp.set_index("Date")["Sales_Revenue"].resample("M").sum().reset_index()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(monthly["Date"], monthly["Sales_Revenue"], marker="o")
            ax.set_xlabel("Date")
            ax.set_ylabel("Total Sales Revenue")
            plt.xticks(rotation=45)
            return fig, ax
        raise ValueError("Required columns for Sales Trend not found.")

    def plot_product_performance():
        if "Product" in df.columns and "Sales_Revenue" in df.columns:
            prod = df.groupby("Product")["Sales_Revenue"].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 4))
            prod.plot(kind="bar", ax=ax)
            ax.set_xlabel("Product")
            ax.set_ylabel("Total Sales Revenue")
            plt.xticks(rotation=0)
            return fig, ax
        raise ValueError("Required columns for Product Performance not found.")

    def plot_regional_analysis():
        if "Region" in df.columns and "Sales_Revenue" in df.columns:
            reg = df.groupby("Region")["Sales_Revenue"].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(reg, labels=reg.index, autopct="%1.1f%%", startangle=90)
            ax.axis("equal")
            return fig, ax
        raise ValueError("Required columns for Regional Analysis not found.")

    safe_plot("Monthly Sales Revenue Trend", plot_sales_trend, "Could not generate Sales Trend.")
    safe_plot("Sales Revenue by Product", plot_product_performance, "Could not generate Product Performance.")
    safe_plot("Sales Distribution by Region", plot_regional_analysis, "Could not generate Regional Analysis.")


# ============================================================
# MAIN UI
# ============================================================

st.title("ContractorsChoice AI Assistant")
st.caption("FastTrack Business Intelligence + Contractor Bidding & Office Assistant")
st.markdown(f"**LLM Backend:** `{LLM_PROVIDER} / {MODEL_NAME}`")

# SIDEBAR — BI DATA
with st.sidebar:
    st.header("BI Data Preparation")
    uploaded_files = st.file_uploader(
        "Upload CSV, Excel, or PDF for BI analysis",
        type=["csv", "xlsx", "xls", "pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        for uf in uploaded_files:
            is_structured = uf.name.endswith((".csv", ".xls", ".xlsx"))
            is_pdf = uf.name.endswith(".pdf")
            if is_structured and uf.name not in st.session_state.dfs:
                handle_file_upload(uf)
            elif is_pdf and uf.name not in st.session_state.pdf_names:
                handle_file_upload(uf)

    st.subheader("Loaded BI Assets")
    if st.session_state.dfs:
        st.success("Structured Files: " + ", ".join(st.session_state.dfs.keys()))
        first_df_name = next(iter(st.session_state.dfs))
        st.dataframe(st.session_state.dfs[first_df_name].head(), use_container_width=True)
    if st.session_state.pdf_names:
        st.success(
            "Unstructured PDF RAG: "
            + ", ".join(st.session_state.pdf_names)
        )

    st.header("Master Agent Status")
    if st.session_state.master_agent_executor:
        st.success("✅ Master BI Agent Ready")
    else:
        st.warning("Upload data to initialize BI Agent.")


tab_bi_chat, tab_bi_visuals, tab_contractors = st.tabs(
    ["📊 BI Analysis & Chat", "📈 BI Visuals", "🛠 Contractor Toolkit"]
)


# ============================================================
# TAB 1: BI ANALYSIS & CHAT
# ============================================================

with tab_bi_chat:
    st.header("Unified AI BI Assistant")
    if st.session_state.master_agent_executor is None:
        st.warning("Upload at least one dataset or PDF to use BI analysis.")
    else:
        for role, text in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(text)

        prompt_hint = "Ask things like: 'Compare product performance and key risks in the PDFs.'"
        user_prompt = st.chat_input(prompt_hint)

        if user_prompt:
            st.session_state.chat_history.append(("user", user_prompt))
            with st.chat_message("user"):
                st.markdown(user_prompt)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing with BI tools..."):
                    try:
                        resp = st.session_state.master_agent_executor.invoke({"input": user_prompt})
                        ans = resp.get("output", str(resp))
                        st.markdown(ans)
                        st.session_state.chat_history.append(("assistant", ans))
                    except Exception as e:
                        st.error(f"BI Agent Error: {e}")
                        st.session_state.chat_history.append(
                            ("assistant", "I hit an error during analysis. Please rephrase or simplify the question.")
                        )


# ============================================================
# TAB 2: BI VISUALS
# ============================================================

with tab_bi_visuals:
    st.header("Automated BI Visualizations")
    if st.session_state.dfs:
        df_names = list(st.session_state.dfs.keys())
        chosen = st.selectbox("Select file to visualize:", df_names)
        df_vis = st.session_state.dfs[chosen]
        st.markdown(f"**Visualizing:** `{chosen}`")
        generate_visual_insights(df_vis)
    else:
        st.warning("Upload structured data (CSV/Excel) to see visualizations.")


# ============================================================
# TAB 3: CONTRACTOR TOOLKIT (Bidding + Office)
# ============================================================

with tab_contractors:
    st.header("🛠 Contractor Toolkit")

    sub_bid, sub_office = st.tabs(["📑 Bid Generator", "📋 Office Assistant"])

    # ----------------------------
    # SUB-TAB: BID GENERATOR
    # ----------------------------
    with sub_bid:
        st.subheader("📑 Smart Bid & Proposal Generator")

        col_client, col_project = st.columns(2)
        with col_client:
            client_name = st.text_input("Client name", key="ct_client_name")
            client_email = st.text_input("Client email (optional)", key="ct_client_email")
            client_phone = st.text_input("Client phone (optional)", key="ct_client_phone")
        with col_project:
            project_name = st.text_input("Project name / short description", key="ct_project_name")
            project_location = st.text_input("Project location / address", key="ct_project_location")
            start_date = st.date_input("Estimated start date", value=date.today(), key="ct_start_date")
            end_date = st.date_input("Estimated completion date", value=date.today(), key="ct_end_date")

        project_description = st.text_area(
            "Project & scope description",
            height=120,
            key="ct_project_desc",
        )

        st.markdown("### Labor & Materials")
        col_labor, col_mat = st.columns(2)
        with col_labor:
            labor_hours = st.number_input(
                "Estimated labor hours",
                min_value=0.0,
                value=40.0,
                step=1.0,
                key="ct_labor_hours",
            )
            labor_rate = st.number_input(
                "Labor rate ($/hour)",
                min_value=0.0,
                value=85.0,
                step=5.0,
                key="ct_labor_rate",
            )
        with col_mat:
            materials_cost = st.number_input(
                "Estimated materials cost ($)",
                min_value=0.0,
                value=2500.0,
                step=100.0,
                key="ct_materials",
            )
            other_costs = st.number_input(
                "Other direct costs ($) (equipment, dumpsters, etc.)",
                min_value=0.0,
                value=0.0,
                step=50.0,
                key="ct_other_costs",
            )

        st.markdown("### Overhead, Profit, Tax")
        col_adj1, col_adj2 = st.columns(2)
        with col_adj1:
            overhead_pct_local = st.number_input(
                "Overhead %",
                min_value=0.0,
                max_value=200.0,
                value=20.0,
                step=1.0,
                key="ct_overhead_pct",
            )
            profit_pct_local = st.number_input(
                "Profit %",
                min_value=0.0,
                max_value=200.0,
                value=15.0,
                step=1.0,
                key="ct_profit_pct",
            )
        with col_adj2:
            tax_pct_local = st.number_input(
                "Sales tax %",
                min_value=0.0,
                max_value=20.0,
                value=0.0,
                step=0.25,
                key="ct_tax_pct",
            )
            contingency_pct = st.number_input(
                "Contingency (% of subtotal)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=1.0,
                key="ct_cont_pct",
            )

        st.markdown("### Payment & Notes")
        payment_terms = st.text_area(
            "Payment terms",
            "50% deposit to schedule work; remaining 50% due upon substantial completion.\n"
            "Change orders billed separately and due upon receipt.",
            height=80,
            key="ct_payment_terms",
        )
        notes = st.text_area("Special notes / exclusions", height=80, key="ct_notes")

        st.markdown("### Job File Uploads (plans, photos, specs)")
        job_files = st.file_uploader(
            "Upload supporting files",
            type=["pdf", "jpg", "jpeg", "png", "doc", "docx"],
            accept_multiple_files=True,
            key="ct_job_files",
        )
        if job_files:
            for uf in job_files:
                st.session_state.contractor_files.append(
                    {"name": uf.name, "size": len(uf.getvalue())}
                )
            st.success(f"Added {len(job_files)} file(s) this session.")
        if st.session_state.contractor_files:
            st.markdown("**Files stored this session:**")
            for f in st.session_state.contractor_files:
                st.write(f"- {f['name']} ({f['size']} bytes)")

        if st.button("⚡ Generate Bid & Proposal", type="primary", key="ct_generate_btn"):
            labor_subtotal = labor_hours * labor_rate
            direct_subtotal = labor_subtotal + materials_cost + other_costs
            overhead_cost = direct_subtotal * (overhead_pct_local / 100.0)
            subtotal_with_overhead = direct_subtotal + overhead_cost
            contingency_amt = subtotal_with_overhead * (contingency_pct / 100.0)
            subtotal_before_profit = subtotal_with_overhead + contingency_amt
            profit_amount = subtotal_before_profit * (profit_pct_local / 100.0)
            pre_tax_total = subtotal_before_profit + profit_amount
            tax_amount = pre_tax_total * (tax_pct_local / 100.0)
            grand_total = pre_tax_total + tax_amount

            st.success("Bid calculated successfully.")

            col_left, col_right = st.columns([1.0, 1.3])
            with col_left:
                st.markdown("#### Cost Breakdown")
                st.write(f"**Labor subtotal:** {money(labor_subtotal)}")
                st.write(f"**Materials:** {money(materials_cost)}")
                st.write(f"**Other direct costs:** {money(other_costs)}")
                st.write(f"**Overhead ({overhead_pct_local:.1f}%):** {money(overhead_cost)}")
                st.write(f"**Contingency ({contingency_pct:.1f}%):** {money(contingency_amt)}")
                st.write(f"**Profit ({profit_pct_local:.1f}%):** {money(profit_amount)}")
                st.write(f"**Pre-tax total:** {money(pre_tax_total)}")
                st.write(f"**Sales tax ({tax_pct_local:.2f}%):** {money(tax_amount)}")
                st.markdown(f"### Grand Total (incl. tax): {money(grand_total)}")

            base_proposal = generate_proposal_text(
                contractor_name="ContractorsChoice Construction, LLC",
                client_name=client_name or "Client",
                project_name=project_name or "Project",
                project_location=project_location or "Project location",
                project_description=project_description,
                start_date=start_date,
                end_date=end_date,
                labor_hours=labor_hours,
                labor_rate=labor_rate,
                materials_cost=materials_cost + other_costs,
                overhead_cost=overhead_cost,
                profit_amount=profit_amount,
                total_price=pre_tax_total,
                payment_terms=payment_terms,
                notes=notes,
            )
            st.session_state.current_proposal = base_proposal

            with col_right:
                st.markdown("#### Proposal (Base Draft)")
                st.text_area(
                    "Proposal (edit before sending if needed)",
                    base_proposal,
                    height=380,
                    key="ct_proposal_text",
                )

        # Smart writer, PDF, SMS
        if st.session_state.current_proposal:
            st.markdown("---")
            st.markdown("### ✨ Smart Proposal Writer (LLM)")
            tone = st.selectbox(
                "Tone for rewrite",
                ["professional", "friendly", "formal", "sales-focused"],
                key="ct_tone_select",
            )
            if st.button("✨ Rewrite with AI", key="ct_smart_rewrite_btn"):
                improved = smart_rewrite_proposal(st.session_state.current_proposal, tone=tone)
                st.text_area(
                    "AI-Improved Proposal",
                    improved,
                    height=380,
                    key="ct_proposal_improved",
                )
                st.session_state.current_proposal = improved

            st.markdown("### 📄 Export as PDF")
            filename_base = (project_name or "proposal").replace(" ", "_")
            pdf_bytes = create_proposal_pdf(st.session_state.current_proposal)
            st.download_button(
                "📥 Download Proposal PDF",
                data=pdf_bytes,
                file_name=f"{filename_base}.pdf",
                mime="application/pdf",
                key="ct_pdf_download_btn",
            )

            st.markdown("### 📲 Send SMS Summary (Twilio)")
            if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER:
                sms_to = st.text_input(
                    "Client mobile number (E.164, e.g. +18015551234)",
                    key="ct_sms_to",
                )
                sms_body = st.text_area(
                    "SMS body (keep short)",
                    f"Hi {client_name or 'there'}, this is ContractorsChoice. "
                    f"We've just sent your detailed proposal for {project_name or 'your project'}. "
                    "Please review it and let us know if you have any questions.",
                    key="ct_sms_body",
                )
                if st.button("Send SMS", key="ct_sms_send_btn"):
                    status = send_sms_via_twilio(sms_to, sms_body)
                    if status.startswith("SMS sent successfully"):
                        st.success(status)
                    else:
                        st.error(status)
            else:
                st.info(
                    "To enable SMS sending, configure TWILIO_ACCOUNT_SID, "
                    "TWILIO_AUTH_TOKEN, and TWILIO_FROM_NUMBER in Streamlit secrets."
                )

    # ----------------------------
    # SUB-TAB: OFFICE ASSISTANT
    # ----------------------------
    with sub_office:
        st.subheader("📋 Office Assistant")

        col_tasks, col_comms = st.columns(2)

        with col_tasks:
            st.markdown("### ✅ Task Tracker")
            new_task = st.text_input("Add a task", key="ct_new_task")
            if st.button("Add Task", key="ct_add_task_btn"):
                if new_task.strip():
                    st.session_state.contractor_tasks.append(
                        {"task": new_task.strip(), "done": False}
                    )

            updated = []
            for i, t in enumerate(st.session_state.contractor_tasks):
                cols = st.columns([0.1, 0.7, 0.2])
                done = cols[0].checkbox("", value=t["done"], key=f"ct_task_done_{i}")
                cols[1].write(t["task"])
                rm = cols[2].button("🗑️", key=f"ct_task_del_{i}")
                if not rm:
                    updated.append({"task": t["task"], "done": done})
            st.session_state.contractor_tasks = updated

            if st.button("Clear completed tasks", key="ct_clear_completed"):
                st.session_state.contractor_tasks = [
                    t for t in st.session_state.contractor_tasks if not t["done"]
                ]

        with col_comms:
            st.markdown("### ✉️ Communication Templates")

            template_type = st.selectbox(
                "Template type",
                [
                    "Select a template...",
                    "Appointment / schedule confirmation",
                    "Estimate sent – follow-up",
                    "Change order approval request",
                    "Payment reminder",
                    "Thank you after job completion",
                ],
                key="ct_template_type",
            )
            tmpl_client = st.text_input("Client name", key="ct_template_client")
            tmpl_project = st.text_input("Project / job name", key="ct_template_project")
            tmpl_amount = st.text_input(
                "Amount (for payment/change order, optional)",
                key="ct_template_amount",
            )

            base_message = ""
            if template_type == "Appointment / schedule confirmation":
                base_message = (
                    f"Hi {tmpl_client or 'there'}, this is ContractorsChoice.\n"
                    f"This is to confirm your appointment for {tmpl_project or 'your project'} on {{DATE/TIME}}.\n"
                    "If you need to reschedule, please reply or call/text this number. Thank you!"
                )
            elif template_type == "Estimate sent – follow-up":
                base_message = (
                    f"Hi {tmpl_client or 'there'}, this is ContractorsChoice.\n"
                    f"Just following up on the estimate we sent for {tmpl_project or 'your project'}.\n"
                    "If you have any questions or would like to discuss options, I'm happy to help."
                )
            elif template_type == "Change order approval request":
                base_message = (
                    f"Hi {tmpl_client or 'there'}, this is ContractorsChoice.\n"
                    f"We've prepared a change order for {tmpl_project or 'your project'} "
                    f"in the amount of {tmpl_amount or '$_____'}.\n"
                    "Please reply with approval or let us know if you have any questions so we can keep your project on schedule."
                )
            elif template_type == "Payment reminder":
                base_message = (
                    f"Hi {tmpl_client or 'there'}, this is ContractorsChoice.\n"
                    f"This is a friendly reminder that a payment of {tmpl_amount or '$_____'} "
                    f"is due for {tmpl_project or 'your project'}.\n"
                    "If you’ve already sent it, please disregard this message. Thank you for your business!"
                )
            elif template_type == "Thank you after job completion":
                base_message = (
                    f"Hi {tmpl_client or 'there'}, this is ContractorsChoice.\n"
                    f"Thank you again for choosing us for {tmpl_project or 'your recent project'}.\n"
                    "If anything comes up or you need additional work, please don’t hesitate to reach out.\n"
                    "We also appreciate any reviews or referrals you’d like to share."
                )

            st.text_area(
                "Message (copy into email/SMS/CRM)",
                value=base_message,
                height=220,
                key="ct_template_message",
            )

        st.markdown("### 🗒️ General Notes")
        st.session_state.contractor_notes = st.text_area(
            "Job / client notes (session only)",
            value=st.session_state.contractor_notes,
            height=150,
            key="ct_notes_global",
        )
