import streamlit as st
import os
########################################################################
# Page 4: Manuals
########################################################################
st.markdown("## Manuals for PredictionAID App")

# Display manuals for each tool (PDFs already present in the directory)
st.markdown("### Manuals (PDF) for Each Tool")

manual_files = {
    "Variable Selection Tool (Page 1)": "manual_VariableSelection.pdf",
    "OLS Tool (Page 2)": "manual_OLS.pdf",
    "Random Forest Tool (Page 3)": "manual_RandomForest.pdf"
}

import base64

for tool, filename in manual_files.items():
    if os.path.exists(filename):
        st.markdown(f"**{tool}:**")
        with open(filename, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf"
            )
            pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
            st.markdown(
                f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="700" height="500" type="application/pdf"></iframe>',
                unsafe_allow_html=True
            )
    else:
        st.warning(f"Manual for {tool} ('{filename}') not found in the directory. Yet to be added")