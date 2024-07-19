import streamlit as st

# Define page content
def introduction_page():
    st.title("Introduction")
    st.write("Welcome to our web app! Here you will find information about our services, articles, and more.")
    st.write("Feel free to explore the pages using the navigation bar.")

def contact_page():
    st.title("Contact Us")
    st.write("Get in touch with us through the following channels:")
    st.write("- Email: example@example.com")
    st.write("- Phone: +1234567890")
    st.write("- Address: 123 Main Street, Anytown, USA")

def articles_page():
    st.title("Articles")
    st.write("Here are some of our latest articles:")
    st.write("- **Article 1**: An overview of our services.")
    st.write("- **Article 2**: Tips and tricks for customer success.")
    st.write("- **Article 3**: How to leverage predictive analytics in your business.")

# Define the navigation bar
def show_navbar():
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Go to", ["Introduction", "Contact", "Articles"])
    
    # Display profile summary and picture
    st.sidebar.image("path/to/profile_picture.jpg", use_column_width=True)
    st.sidebar.write("**Name:** Your Name")
    st.sidebar.write("**Position:** Your Position")
    st.sidebar.write("**Bio:** Brief bio or description.")
    
    # Download resume button
    st.sidebar.download_button(
        label="Download Resume",
        data=open("path/to/resume.pdf", "rb").read(),
        file_name="resume.pdf",
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
