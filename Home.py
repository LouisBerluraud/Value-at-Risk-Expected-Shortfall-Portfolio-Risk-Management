import streamlit as st

# Setting the description and the title
st.title('Optimize your Portfolio')
st.text("""Welcome! Here's what you’ll discover:
        
1. Value at Risk Models – Learn how to modelize VaR and compare 
one model to another.""")
st.page_link("pages/1_VaR Models.py", label="📊 VaR Models")

st.text("""2. Rockafellar Optimization – Explore how CVaR can be used 
to optimize your portfolio.""")
st.page_link("pages/2_Rockafellar Optimization.py", label="📈 Rockafellar Optimization")


st.write("### Collaborators of the Project")
st.markdown(
    """
    <div style='margin-bottom: 25px; padding: 15px; background-color: #ebebeb; border-radius: 15px;'>
        <span style='font-weight: bold; color: #0A66C2; font-size: 24px;'>Created by:</span><br>
        <a href='https://www.linkedin.com/in/louis-berluraud-a41098204/' target='_blank' style='text-decoration: none; display: flex; align-items: center; gap: 12px; margin-top: 8px;'>
            <img src='https://cdn-icons-png.flaticon.com/512/174/174857.png' width='32' height='32'/>
            <span style='color: #0A66C2; font-size: 18px; font-weight: bold;'>Louis Berluraud</span>
        <a href='https://www.linkedin.com/in/guillaume-thiebaut-88a137283/' target='_blank' style='text-decoration: none; display: flex; align-items: center; gap: 12px; margin-top: 8px;'>
            <img src='https://cdn-icons-png.flaticon.com/512/174/174857.png' width='32' height='32'/>
            <span style='color: #0A66C2; font-size: 18px; font-weight: bold;'>Guillaume Thiebaut</span>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)