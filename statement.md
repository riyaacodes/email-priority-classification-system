# Project Statement

## 1. Problem Statement
Organizations receive a large volume of emails every day, ranging from critical operational issues to routine messages. Manually identifying which emails require immediate attention is time-consuming and may lead to delays in responding to urgent communication. Keyword-based filtering is not reliable because urgency depends on context and meaning rather than specific words.

There is a need for an intelligent system that can automatically analyze email content and classify messages into different priority levels to help users focus on what matters first.

---

## 2. Scope of the Project
This project focuses on developing an Email Priority Classification System that:

- Processes email subject and body text from a CSV file  
- Classifies each email into **Urgent**, **Normal**, or **Low Priority**
- Provides an interactive interface for uploading and viewing results
- Generates a downloadable output CSV with predicted labels

The project does **not** include:
- Integration with live email services (e.g., Gmail/Outlook APIs)
- Storage in a database
- Real-time notification or automation workflows

These may be considered in future enhancements.

---

## 3. Target Users
The system is designed for:

- Employees handling large volumes of internal communication
- IT helpdesk and support teams
- Students and academic users learning ML concepts
- Anyone who needs quick prioritization of email datasets

---

## 4. High-Level Features
- Upload or select CSV datasets through a Streamlit dashboard
- Automatic preprocessing of email subject and body
- Semantic text embedding using MiniLM (pre-trained transformer model)
- Urgency prediction using a Logistic Regression classifier
- Display of classified results in a table
- Visualization of urgency distribution using a bar chart
- Downloadable output file with predictions in csv format
