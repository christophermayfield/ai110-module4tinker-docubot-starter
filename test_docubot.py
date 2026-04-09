from docubot import DocuBot

def test_retrieval():
    bot = DocuBot(docs_folder="docs")
    print(f"Loaded {len(bot.documents)} documents.")
    
    # Check index for a known word
    query = "database"
    print(f"\nQuerying: '{query}'")
    results = bot.retrieve(query, top_k=1)
    
    if results:
        filename, text = results[0]
        print(f"Top match: {filename}")
        print(f"Snippet start: {text[:100]}...")
    else:
        print("No results found.")

if __name__ == "__main__":
    test_retrieval()
