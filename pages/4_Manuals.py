import streamlit as st
import os
import base64
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

for tool, filename in manual_files.items():
    if os.path.exists(filename):
        st.markdown(f"**{tool}:**")
        
        # Read PDF file only once
        with open(filename, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Download button
            st.download_button(
                label="📥 Download PDF",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf",
                key=f"download_{filename}"  # Unique key for each button
            )
        
        # Show file size info
        file_size = len(pdf_bytes) / 1024  # KB
        st.caption(f"📄 File size: {file_size:.1f} KB")
        
        st.markdown("---")
    else:
        st.warning(f"Manual for {tool} ('{filename}') not found in the directory. Yet to be added")