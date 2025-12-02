
import streamlit as st
from datetime import date

st.set_page_config(page_title="Contractor AI Toolkit", layout="wide")

def money(x: float) -> str:
    return f"${x:,.2f}"

def generate_proposal_text(
    contractor_name, client_name, project_name, project_location,
    project_description, start_date, end_date, labor_hours,
    labor_rate, materials_cost, overhead_cost, profit_amount,
    total_price, payment_terms, notes,
):
    start = start_date.strftime("%B %d, %Y")
    end = end_date.strftime("%B %d, %Y")

    text = f"""
{contractor_name}
Proposal for: {client_name}
Project: {project_name}
Location: {project_location}

1. Project Overview
{project_description}

Start: {start}
End: {end}

2. Pricing Summary
Labor: {money(labor_hours * labor_rate)}
Materials: {money(materials_cost)}
Overhead: {money(overhead_cost)}
Profit: {money(profit_amount)}

Total price: {money(total_price)}

Payment Terms:
{payment_terms}

Notes:
{notes}
"""
    return text

st.title("🛠 Contractor AI Toolkit")
tab_bid, tab_office = st.tabs(["📑 Bid Generator", "📋 Office Assistant"])

with tab_bid:
    st.subheader("📑 Smart Bid Generator")

    client_name = st.text_input("Client Name")
    project_name = st.text_input("Project Name")
    project_location = st.text_input("Location")
    project_description = st.text_area("Project Description")
    start_date = st.date_input("Start Date", value=date.today())
    end_date = st.date_input("End Date", value=date.today())

    labor_hours = st.number_input("Labor Hours", value=20.0)
    labor_rate = st.number_input("Labor Rate ($/hr)", value=85.0)
    materials_cost = st.number_input("Materials Cost ($)", value=1000.0)

    overhead_pct = st.number_input("Overhead (%)", value=20.0)
    profit_pct = st.number_input("Profit (%)", value=15.0)

    payment_terms = st.text_area("Payment Terms", "50% upfront, 50% on completion.")
    notes = st.text_area("Notes")

    if st.button("Generate Proposal"):
        labor_total = labor_hours * labor_rate
        overhead_cost = (labor_total + materials_cost) * (overhead_pct / 100)
        profit_amount = (labor_total + materials_cost + overhead_cost) * (profit_pct / 100)
        total_price = labor_total + materials_cost + overhead_cost + profit_amount

        proposal = generate_proposal_text(
            "Acme Construction",
            client_name,
            project_name,
            project_location,
            project_description,
            start_date,
            end_date,
            labor_hours,
            labor_rate,
            materials_cost,
            overhead_cost,
            profit_amount,
            total_price,
            payment_terms,
            notes
        )

        st.success("Proposal Generated!")
        st.text_area("Proposal", proposal, height=300)

with tab_office:
    st.subheader("📋 Office Assistant")

    if "tasks" not in st.session_state:
        st.session_state.tasks = []

    new_task = st.text_input("New Task")
    if st.button("Add Task"):
        st.session_state.tasks.append(new_task)

    for t in st.session_state.tasks:
        st.write("• " + t)
