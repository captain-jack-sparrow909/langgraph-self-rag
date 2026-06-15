from graph.graph import app

def main():
    print("Hello from langgraph-agentic-rag!")
    print(app.invoke(input={"question": "what is agent memory?"}))


if __name__ == "__main__":
    main()
