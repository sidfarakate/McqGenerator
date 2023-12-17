import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.McqGenerator.utils import read_file, get_table_data
import streamlit as sl
from langchain.callbacks import get_openai_callback
from src.McqGenerator.mcqgen import generate_evaluate_chain
from src.McqGenerator.logger import logging

# Loading the JSON file
with open(r"C:\Users\SVF\McqGenerator\Response.json", 'r') as file:
    RESPONSE_JSON = json.load(file)

# Creating the title for the app
sl.title("MCQs Generation Application")

# Create a form using st.form
with sl.form("user_inputs"):
    # File Upload
    uploaded_file = sl.file_uploader("Upload a PDF or Text file")

    # Input field
    mcq_count = sl.number_input("Number of MCQs", min_value = 3, max_value = 50)

    # Subject
    subject = sl.text_input("Insert Subject", max_chars = 20)

    # Quiz Tone
    tone = sl.text_input("Complexity level of Questions", max_chars= 20, placeholder= 'simple')

    # Add button
    button = sl.form_submit_button("Create MCQs")

    # Check if the button is clicked and all fields have input
    if button and uploaded_file is not None and mcq_count and subject and tone:
        with sl.spinner("Loading..."):
            try:
                text = read_file(uploaded_file)

                # Count token and the cost of API call
                with get_openai_callback() as cb:
                    response=generate_evaluate_chain(
                    {
                        "text": text,
                        "number": mcq_count,
                        "subject": subject,
                        "tone": tone,
                        "response_json": json.dumps(RESPONSE_JSON)
                    })

            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                sl.error("Error")

            else:
                print("Total Tokens: ", cb.total_tokens)
                print("Prompt Tokens: ", cb.prompt_tokens)
                print("Completion Tokens: ", cb.completion_tokens)
                print("Total Cost: ", cb.total_cost)

                if isinstance(response, dict):
                    # Extract the quiz data from the response
                    quiz = response.get('quiz', None)
                    if quiz is not None:
                        table_data = get_table_data(quiz)
                        if table_data is not None:
                            df = pd.DataFrame(table_data)
                            df.index = df.index + 1
                            sl.table(df)

                            # Display the review in a text box as well
                            sl.text_area(label='Review', value= response['review'])

                        else:
                            sl.error("Error in the table data")

                else:
                    sl.write(response)

