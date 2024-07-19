import streamlit as st
import requests
# Define page content
def introduction_page():
    st.title("Introduction")
    st.write("Welcome to our web app! Here you will find information about our services, articles, and more.")
    st.write("Feel free to explore the pages using the navigation bar.")

def contact_page():
    st.title("Contact Us")
    st.write("Get in touch with us through the following channels:")
    st.write("- Email: sohamnsharma@gmail.com")
    st.write("- Phone: +91-9730084626")
    st.write("- Address: Indore MP")

def articles_page():
    st.title("Articles")
    st.write("Here are some of our latest articles:")
    st.write("- **Article 1**: An overview of our services.")
    st.write("- **Article 2**: Tips and tricks for customer success.")
    st.write("- **Article 3**: How to leverage predictive analytics in your business.")


#Nav Bar
def show_navbar():
    st.sidebar.title("Navigation")
    
    # Display profile summary and picture
    profile_image_url = "https://github.com/soham2710/Customer_Success_Dashboard/raw/main/BH6A0835.jpg"
    st.sidebar.image(profile_image_url, use_column_width=True)
    st.sidebar.write("**Name:** Your Name")
    st.sidebar.write("**Position:** Your Position")
    st.sidebar.write("**Bio:** Brief bio or description.")
    
   # Fetch resume from URL
    resume_url = "https://github.com/soham2710/Customer_Success_Dashboard/blob/main/Customer%20Success%20Resume.pdf?raw=true"
    response = requests.get(resume_url)
    
    st.sidebar.download_button(
        label="Download Resume",
        data=response.content,
        file_name="Customer Success Resume.pdf",
        mime="application/pdf"
    )
    
    return selected_page

# Main function
def main():
    st.set_page_config(page_title="CSM Dashboard", page_icon=":guardsman:", layout="wide")
    
    selected_page = show_navbar()
    
    if selected_page == "Introduction":
        introduction_page()
    elif selected_page == "Contact":
        contact_page()
    elif selected_page == "Articles":
        articles_page()

if __name__ == "__main__":
    main()
