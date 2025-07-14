
from transformers import pipeline


def ai():
    question = input("Enter your question: ")
    pipe = pipeline("text2text-generation", model="google/flan-t5-base")
    input_text = f"Question : {question}"
    result = pipe(input_text, max_length=150, do_sample=False, temperature=0.8, top_p=0.9)
    print("Answer :", result[0]['generated_text'])

if __name__ == "__main__":
    ai()