from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import traceback
import json
import yfinance as yf
import numpy as np
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple vector store setup
def setup_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",  # Smaller model
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
def setup_vectordb(embeddings, persist_directory="./data/new_vector_db"):
    from langchain_community.vectorstores import Chroma
    import os
    import shutil
    
    # Option 1: Try to create the directory with exist_ok
    os.makedirs(persist_directory, exist_ok=True)
    
    try:
        # Add client settings to handle the directory creation properly
        from chromadb.config import Settings
        client_settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory,
            anonymized_telemetry=False
        )
        
        # Try creating the Chroma instance with explicit client settings
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            client_settings=client_settings
        )
    except Exception as e:
        # If that fails, try a more direct approach - delete the directory and recreate
        print(f"First attempt failed: {e}")
        try:
            # Create a direct ChromaDB client
            import chromadb
            client = chromadb.PersistentClient(path=persist_directory)
            # Use the client to create the Chroma instance
            return Chroma(
                client=client,
                embedding_function=embeddings
            )
        except Exception as e2:
            print(f"Second attempt failed: {e2}")
            # As a last resort, we'll remove and recreate the directory
            print(f"Removing existing directory at {persist_directory}")
            shutil.rmtree(persist_directory, ignore_errors=True)
            os.makedirs(persist_directory, exist_ok=True)
            return Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )

# Add this class definition to your main.py file before the endpoints
class VectorStoreManager:
    def __init__(self):
        """Initialize the vector store manager"""
        self.embeddings = setup_embeddings()
        self.vectordb = setup_vectordb(self.embeddings)
    
    def clear_collection(self, ticker):
        """Clear only documents for a specific ticker"""
        try:
            if hasattr(self.vectordb, "_collection"):
            # Only clear documents for this specific ticker
                self.vectordb._collection.delete(
                    where={"ticker": ticker}
                )
                return True
            return False
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False
    
    def add_documents(self, documents, metadata):
        """Add documents to the vector store"""
        if not documents:
            return False
        
        texts = documents
        metadatas = metadata
        
        # Add to vector database
        self.vectordb.add_texts(texts=texts, metadatas=metadatas)
        #self.vectordb.persist()
        
        print(f"Added {len(texts)} documents to vector database")
        return True

class DataIngestor:
    def __init__(self):
        """Initialize data ingestor with vector store manager"""
        self.vector_store_manager = VectorStoreManager()

    def fetch_stock_data(self, ticker: str, period: str = "1y"):
        """Fetch stock data for a given ticker"""
        try:
            # Fetch stock information
            stock = yf.Ticker(ticker)

            # Get historical market data
            hist = stock.history(period=period)

            # Get company info
            info = stock.info

            # Get financial statements
            financials = {
                "income_statement": stock.financials,
                "balance_sheet": stock.balance_sheet,
                "cashflow": stock.cashflow
            }

            return {
                "ticker": ticker,
                "historical_data": hist.to_dict(),
                "company_info": info,
                "financials": {k: v.to_dict() for k, v in financials.items()}
            }
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return {}

    def prepare_document_text(self, stock_data):
        """Prepare document text from stock data for vector store"""
        if not stock_data:
            return []

        documents = []

        # Company overview
        company_doc = f"""
        Stock Ticker: {stock_data.get('ticker', 'N/A')}
        
        Company Overview:
        - Name: {stock_data.get('company_info', {}).get('longName', 'N/A')}
        - Sector: {stock_data.get('company_info', {}).get('sector', 'N/A')}
        - Industry: {stock_data.get('company_info', {}).get('industry', 'N/A')}
        - Description: {stock_data.get('company_info', {}).get('longBusinessSummary', 'No description available')}
        """
        documents.append(company_doc)

        # Historical price summary
        hist_data = stock_data.get('historical_data', {})
        if hist_data and 'Close' in hist_data:
            # Convert dictionary values to a list and then calculate statistics
            close_values = list(hist_data['Close'].values())
            high_values = list(hist_data['High'].values()) if 'High' in hist_data else []
            low_values = list(hist_data['Low'].values()) if 'Low' in hist_data else []
            
            avg_close = sum(close_values) / len(close_values) if close_values else 'N/A'
            max_high = max(high_values) if high_values else 'N/A'
            min_low = min(low_values) if low_values else 'N/A'
            
            price_doc = f"""
            Historical Price Summary:
            - Average Close Price: {avg_close if avg_close != 'N/A' else 'N/A'}
            - Highest Price: {max_high if max_high != 'N/A' else 'N/A'}
            - Lowest Price: {min_low if min_low != 'N/A' else 'N/A'}
            """
            documents.append(price_doc)

        # Financial statement summary
        financials = stock_data.get('financials', {})
        for statement_type, statement_data in financials.items():
            if statement_data:
                fin_doc = f"""
                {statement_type.replace('_', ' ').title()} Summary:
                """
                # Get first 5 items, handling nested dictionary structure
                items = list(statement_data.items())[:5]
                for key, value in items:
                    fin_doc += f"- {key}: {value}\n"
                documents.append(fin_doc)

        return documents

    def fetch_and_ingest_all_data(self, ticker: str):
        """Fetch stock data and ingest into vector store"""
        # Clear existing collection for this ticker
        self.vector_store_manager.clear_collection(ticker)

        # Fetch stock data
        stock_data = self.fetch_stock_data(ticker)

        # Prepare document texts
        documents = self.prepare_document_text(stock_data)

        # Prepare metadata
        metadata = [{"ticker": ticker, "type": "stock_info"} for _ in documents]

        # Add documents to vector store
        if documents:
            return self.vector_store_manager.add_documents(documents, metadata)
        return False

# Initialize components at startup
@app.on_event("startup")
async def startup_event():
    print("Initializing components...")
    try:
        # Initialize embeddings and vector store
        embeddings = setup_embeddings()
        vectordb = setup_vectordb(embeddings)
        print("✅ Components initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing components: {e}")

# Root endpoint to display all routes
@app.get("/")
async def root():
    routes = []
    for route in app.routes:
        routes.append({
            "path": route.path,
            "name": route.name,
            "methods": route.methods
        })
    return {"available_routes": routes}

# Test endpoint
@app.get("/test")
async def test():
    return {"message": "Test endpoint working!"}

# Define request models
class IngestDataRequest(BaseModel):
    ticker: str

class RAGQueryRequest(BaseModel):
    question: str
    ticker: str = None

class AgentQueryRequest(BaseModel):
    query: str

class PortfolioOptimizationRequest(BaseModel):
    tickers: list[str]
    allocations: dict[str, float]
    risk_preference: str

# Create the ingest_stock_data endpoint
@app.post("/rag/ingest_stock_data")
async def ingest_stock_data(request: IngestDataRequest):
    """Ingest data for a specific stock into the RAG system"""
    try:
        # Create a new DataIngestor instance
        data_ingester = DataIngestor()
        
        print(f"Ingesting data for ticker: {request.ticker}")
        success = data_ingester.fetch_and_ingest_all_data(request.ticker)
        print(f"Ingestion result: {success}")
        return {"success": success, "ticker": request.ticker}
    except Exception as e:
        print(f"Error ingesting data: {e}")
        print(traceback.format_exc())
        return {"success": False, "error": str(e), "ticker": request.ticker}

# Regular query endpoint
@app.post("/rag/query")
async def rag_query(request: RAGQueryRequest):
    """Query the RAG system for stock information"""
    try:
        # Initialize components
        embeddings = setup_embeddings()
        vectordb = setup_vectordb(embeddings)
        
        # Format the query
        query = request.question
        if request.ticker:
            query = f"{request.ticker}: {request.question}"
        
        # Search the vector store
        docs = vectordb.similarity_search(query, k=2)
        
        # Extract content from docs
        context = "\n\n".join([doc.page_content for doc in docs])
        sources = [doc.metadata for doc in docs]
        
        # Create a prompt for Groq
        prompt = f"""
        You are a financial expert assistant. Use the following information to answer the question.
        If the information doesn't contain the answer, say that you don't have enough information.
        
        Information:
        {context}
        
        Question: {request.question} about {request.ticker if request.ticker else 'the company'}.
        
        Provide a well-structured, detailed analysis based on the information. Include specific 
        numbers and data points where available. Format the response using markdown for readability.
        """
        
        # Call Groq API
        answer = call_groq_api(prompt)
        
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        print(f"Error querying RAG system: {e}")
        print(traceback.format_exc())
        return {"error": str(e)}

def call_groq_api(prompt, model="llama3-70b-8192"):
    """Call Groq API directly"""
    import requests
    
    GROQ_API_KEY = "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"
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a financial expert assistant providing detailed stock and financial analysis."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 800
    }
    
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload
    )
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print(f"Error from Groq API: {response.status_code} - {response.text}")
        return f"Error generating response. API returned status code {response.status_code}."

# Agent query endpoint
@app.post("/rag/agent_query")
async def agent_query(request: AgentQueryRequest):
    """Run the stock agent with a user query"""
    try:
        # Initialize components
        embeddings = setup_embeddings()
        vectordb = setup_vectordb(embeddings)
        
        # Get relevant documents from the database
        docs = vectordb.similarity_search(request.query, k=2)
        # Limit context length
        max_tokens = 2000
        context = "\n\n".join([doc.page_content for doc in docs])
        if len(context) > max_tokens:
            context = context[:max_tokens] + "..."
        
        # Extract ticker symbols from the query
        import re
        tickers = re.findall(r'\b[A-Z]{1,5}\b', request.query)
        ticker_info = ""
        
        # If we have ticker symbols, get current data for them
        if tickers:
            import yfinance as yf
            ticker_info = "Current market data:\n"
            
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    price = stock.history(period="1d")["Close"].iloc[-1]
                    info = stock.info
                    
                    ticker_info += f"- {ticker}: ${price:.2f}\n"
                    ticker_info += f"  P/E Ratio: {info.get('trailingPE', 'N/A')}\n"
                    ticker_info += f"  Market Cap: ${info.get('marketCap', 0)/1e9:.2f}B\n"
                except Exception as e:
                    ticker_info += f"- {ticker}: Error retrieving data: {str(e)}\n"
        
        # Create a combined prompt
        prompt = f"""
        You are a sophisticated stock analysis agent that helps users with financial questions.
        
        User query: {request.query}
        
        Relevant information from my knowledge base:
        {context}
        
        {ticker_info if ticker_info else ""}
        
        Provide a detailed, well-structured response to the user's query. Use specific data points and 
        numbers where available. If you're analyzing a stock, include both fundamental and technical 
        factors. Format your response using markdown for readability.
        """
        
        # Call Groq API
        response = call_groq_api(prompt)
        
        return {"result": response}
    except Exception as e:
        print(f"Error in agent query: {e}")
        print(traceback.format_exc())
        return {"error": str(e)}
# Comprehensive analysis endpoint
@app.get("/rag/comprehensive_analysis/{ticker}")
async def comprehensive_analysis(ticker: str):
    """Perform a comprehensive analysis of a stock"""
    try:
        # Initialize components
        embeddings = setup_embeddings()
        vectordb = setup_vectordb(embeddings)
        
        # Ensure data is ingested
        data_ingester = DataIngestor()
        data_ingester.fetch_and_ingest_all_data(ticker)
        
        # Get stock data
        stock_data = data_ingester.fetch_stock_data(ticker)
        
        # Get relevant docs from vector store for fundamental analysis
        docs = vectordb.similarity_search(f"{ticker} company financials", k=3)
        fundamental_context = "\n\n".join([doc.page_content for doc in docs])
        
        # Get live data using yfinance for technical analysis
        import yfinance as yf
        import numpy as np
        
        stock = yf.Ticker(ticker)
        hist = stock.history(period="200d")  # Get 200 days of data for technical analysis
        
        # Calculate technical indicators
        # Simple Moving Averages
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
        
        # RSI - Relative Strength Index
        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD - Moving Average Convergence Divergence
        hist['EMA_12'] = hist['Close'].ewm(span=12, adjust=False).mean()
        hist['EMA_26'] = hist['Close'].ewm(span=26, adjust=False).mean()
        hist['MACD'] = hist['EMA_12'] - hist['EMA_26']
        hist['Signal_Line'] = hist['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        hist['Middle_Band'] = hist['Close'].rolling(window=20).mean()
        hist['STD'] = hist['Close'].rolling(window=20).std()
        hist['Upper_Band'] = hist['Middle_Band'] + (hist['STD'] * 2)
        hist['Lower_Band'] = hist['Middle_Band'] - (hist['STD'] * 2)
        
        # Get current values for technical indicators
        current_price = hist['Close'].iloc[-1]
        current_sma_50 = hist['SMA_50'].iloc[-1]
        current_sma_200 = hist['SMA_200'].iloc[-1]
        current_rsi = hist['RSI'].iloc[-1]
        current_macd = hist['MACD'].iloc[-1]
        current_signal = hist['Signal_Line'].iloc[-1]
        current_upper_band = hist['Upper_Band'].iloc[-1]
        current_middle_band = hist['Middle_Band'].iloc[-1]
        current_lower_band = hist['Lower_Band'].iloc[-1]
        
        # Determine support and resistance levels (simplified)
        recent_lows = hist['Low'].tail(30).nsmallest(3).mean()
        recent_highs = hist['High'].tail(30).nlargest(3).mean()
        
        # Prepare technical data
        technical_data = {
            "Current Price": current_price,
            "SMA_50": current_sma_50,
            "SMA_200": current_sma_200,
            "RSI_14": current_rsi,
            "MACD": current_macd,
            "Signal_Line": current_signal,
            "Upper_Band": current_upper_band,
            "Middle_Band": current_middle_band,
            "Lower_Band": current_lower_band,
            "Support_Level": recent_lows,
            "Resistance_Level": recent_highs
        }
        
        # Create separate prompts for fundamental and technical analysis
        fundamental_prompt = f"""
        Provide a fundamental analysis of {ticker} based on the following information.
        
        Company information and financial data:
        {fundamental_context}
        
        Focus on business model, financial performance, growth prospects, and valuation metrics.
        Format your response using markdown for readability.
        Keep your analysis concise and focused on fundamental factors only.
        """
        
        technical_prompt = f"""
        Provide a technical analysis of {ticker} based on the following technical indicators:
        
        Current Price: ${technical_data['Current Price']:.2f}
        50-Day Moving Average: ${technical_data['SMA_50']:.2f}
        200-Day Moving Average: ${technical_data['SMA_200']:.2f}
        Relative Strength Index (RSI-14): {technical_data['RSI_14']:.2f}
        MACD: {technical_data['MACD']:.4f}
        MACD Signal Line: {technical_data['Signal_Line']:.4f}
        Bollinger Bands:
        - Upper Band: ${technical_data['Upper_Band']:.2f}
        - Middle Band: ${technical_data['Middle_Band']:.2f}
        - Lower Band: ${technical_data['Lower_Band']:.2f}
        Support Level: ${technical_data['Support_Level']:.2f}
        Resistance Level: ${technical_data['Resistance_Level']:.2f}
        
        Focus ONLY on technical analysis patterns, price movements, and trading signals.
        Identify the trend direction, support/resistance levels, and any actionable signals.
        Do NOT include fundamental analysis or company information.
        Format your response using markdown for readability.
        """
        
        # Get separate analyses from LLM
        fundamental_analysis = call_groq_api(fundamental_prompt)
        technical_analysis = call_groq_api(technical_prompt)
        
        # Create combined prompt for overall recommendation
        outlook_prompt = f"""
        Based on both fundamental and technical analysis, provide an overall outlook and recommendation for {ticker}.
        
        Fundamental Analysis Summary:
        {fundamental_analysis[:500]}...
        
        Technical Analysis Summary:
        {technical_analysis[:500]}...
        
        Provide a concise outlook and clear investment recommendation (Buy, Hold, or Sell).
        Include a target price range and key risks to watch.
        Format your response using markdown for readability.
        """
        
        # Get overall recommendation
        outlook = call_groq_api(outlook_prompt)
        
        # Extract basic information for structured response
        company_name = stock_data.get("company_info", {}).get("longName", ticker)
        sector = stock_data.get("company_info", {}).get("sector", "N/A")
        market_cap = stock_data.get("company_info", {}).get("marketCap", 0)
        pe_ratio = stock_data.get("company_info", {}).get("trailingPE", "N/A")
        eps = stock_data.get("company_info", {}).get("trailingEps", "N/A")
        
        # Create structured response
        response = {
            "stock_info": {
                "name": company_name,
                "symbol": ticker,
                "sector": sector,
                "price": current_price,
                "market_cap": market_cap
            },
            "fundamental_analysis": {
                "metrics": {
                    "PE Ratio": pe_ratio,
                    "EPS": eps,
                },
                "analysis": fundamental_analysis
            },
            "technical_analysis": {
                "indicators": technical_data,
                "analysis": technical_analysis
            },
            "rag_insights": {
                "answer": outlook,
                "sources": [doc.metadata for doc in docs]
            }
        }
        
        return response
    except Exception as e:
        print(f"Error in comprehensive analysis: {e}")
        print(traceback.format_exc())
        return {"error": str(e)}
    
@app.get("/rag/list_documents/{ticker}")
async def list_documents(ticker: str):
    """List documents in the vector database for a specific ticker"""
    try:
        # Initialize components
        embeddings = setup_embeddings()
        vectordb = setup_vectordb(embeddings)
        
        # Query for documents with the ticker
        results = vectordb.similarity_search(ticker, k=20)
        
        # Format the results
        documents = []
        for i, doc in enumerate(results):
            documents.append({
                "id": i,
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return {"documents": documents, "count": len(documents)}
    except Exception as e:
        print(f"Error listing documents: {e}")
        print(traceback.format_exc())
        return {"error": str(e)}
# Portfolio optimization endpoint
@app.post("/rag/optimize_portfolio")
async def optimize_portfolio(request: PortfolioOptimizationRequest):
    """Optimize a portfolio using the RAG agent"""
    try:
        # Initialize components
        embeddings = setup_embeddings()
        vectordb = setup_vectordb(embeddings)
        
        # Gather data for each ticker
        tickers_data = {}
        for ticker in request.tickers:
            # Get data from vector database
            docs = vectordb.similarity_search(f"{ticker} analysis", k=1)
            ticker_context = "\n\n".join([doc.page_content for doc in docs])
            
            # Get live data
            try:
                stock = yf.Ticker(ticker)
                current_data = {
                    "price": stock.history(period="1d")["Close"].iloc[-1],
                    "pe_ratio": stock.info.get("trailingPE", "N/A"),
                    "beta": stock.info.get("beta", 1.0),
                    "market_cap": stock.info.get("marketCap", 0),
                    "dividend_yield": stock.info.get("dividendYield", 0)
                }
                
                # Calculate risk and return metrics
                hist = stock.history(period="1y")
                if not hist.empty:
                    returns = hist["Close"].pct_change().dropna()
                    current_data["avg_return"] = returns.mean() * 252 * 100  # Annualized
                    current_data["volatility"] = returns.std() * np.sqrt(252) * 100  # Annualized
                
                tickers_data[ticker] = {
                    "context": ticker_context,
                    "current_data": current_data
                }
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                tickers_data[ticker] = {
                    "context": ticker_context,
                    "current_data": {"error": str(e)}
                }
        
        # Create a prompt for portfolio optimization
        portfolio_context = "\n\n".join([
            f"--- {ticker} ---\n{data['context']}\nCurrent Data: {json.dumps(data['current_data'])}"
            for ticker, data in tickers_data.items()
        ])
        
        # Truncate if too long
        max_tokens = 3000
        if len(portfolio_context) > max_tokens:
            portfolio_context = portfolio_context[:max_tokens] + "..."
        
        prompt = f"""
        As a portfolio optimization expert, optimize the following portfolio based on a {request.risk_preference} risk profile:
        
        Tickers: {', '.join(request.tickers)}
        Current allocations: {json.dumps(request.allocations)}
        Risk preference: {request.risk_preference} (low, medium, or high)
        
        Stock information:
        {portfolio_context}
        
        Please optimize this portfolio considering:
        1. The risk profile ({request.risk_preference})
        2. Diversification principles
        3. Current market conditions
        4. Each stock's historical performance and metrics
        
        First, analyze each stock's strengths, weaknesses, and fit with the risk profile.
        Then, provide an optimized allocation with clear justification for each weight.
        
        IMPORTANT: Your final allocation recommendations MUST be presented in a clear table format with the exact format:
        | Ticker | Optimized Allocation |
        | --- | --- |
        | TICKER1 | XX% |
        | TICKER2 | YY% |
        
        Where the percentages must sum to exactly 100%.
        
        Return your analysis in markdown format, followed by your allocation table and justification.
        """
        
        # Call LLM for portfolio optimization
        analysis = call_groq_api(prompt)
        
        # Try to extract the LLM's allocation recommendations first
        try:
            import re
            # Look for a table or list format in the analysis
            allocation_pattern = r'([A-Z]{1,5})\s*\|\s*(\d+\.?\d*)%'
            matches = re.findall(allocation_pattern, analysis)
            
            llm_portfolio = {}
            if matches:
                for ticker, allocation in matches:
                    if ticker in request.tickers:  # Only include tickers from the request
                        llm_portfolio[ticker] = float(allocation)
                
                # Check if all requested tickers are included and allocations sum close to 100%
                if (all(ticker in llm_portfolio for ticker in request.tickers) and 
                    abs(sum(llm_portfolio.values()) - 100) < 10):
                    
                    # Normalize to exactly 100%
                    total = sum(llm_portfolio.values())
                    optimized_portfolio = {
                        ticker: round((alloc / total) * 100, 2) 
                        for ticker, alloc in llm_portfolio.items()
                    }
                else:
                    # If allocations don't sum to ~100% or not all tickers found, 
                    # use calculated allocations
                    optimized_portfolio = calculate_portfolio_allocation(
                        tickers_data, request.tickers, request.risk_preference
                    )
            else:
                # If we didn't find any matches, try a different pattern
                allocation_pattern = r'([A-Z]{1,5})(?:[:\s-]+)(\d+\.?\d*)%'
                matches = re.findall(allocation_pattern, analysis)
                
                if matches:
                    for ticker, allocation in matches:
                        if ticker in request.tickers:
                            llm_portfolio[ticker] = float(allocation)
                    
                    if (all(ticker in llm_portfolio for ticker in request.tickers) and 
                        abs(sum(llm_portfolio.values()) - 100) < 10):
                        
                        total = sum(llm_portfolio.values())
                        optimized_portfolio = {
                            ticker: round((alloc / total) * 100, 2) 
                            for ticker, alloc in llm_portfolio.items()
                        }
                    else:
                        optimized_portfolio = calculate_portfolio_allocation(
                            tickers_data, request.tickers, request.risk_preference
                        )
                else:
                    # If no pattern matches, use calculated allocations
                    optimized_portfolio = calculate_portfolio_allocation(
                        tickers_data, request.tickers, request.risk_preference
                    )
        except Exception as e:
            print(f"Error extracting allocations from LLM response: {e}")
            print(traceback.format_exc())
            # Use calculated portfolio as fallback
            optimized_portfolio = calculate_portfolio_allocation(
                tickers_data, request.tickers, request.risk_preference
            )
        
        return {
            "optimized_portfolio": optimized_portfolio,
            "analysis": analysis
        }
    except Exception as e:
        print(f"Error optimizing portfolio: {e}")
        print(traceback.format_exc())
        return {"error": str(e)}

def calculate_portfolio_allocation(tickers_data, tickers, risk_preference):
    """Calculate portfolio allocation based on risk preference and stock metrics"""
    import numpy as np
    
    optimized_portfolio = {}
    total_weight = 0
    
    for ticker in tickers:
        if ticker in tickers_data and "current_data" in tickers_data[ticker]:
            current_data = tickers_data[ticker]["current_data"]
            
            # Get metrics for optimization (with defaults if missing)
            volatility = current_data.get("volatility", 20)
            avg_return = current_data.get("avg_return", 10)
            beta = current_data.get("beta", 1.0)
            
            # Adjust weight based on risk preference
            if risk_preference == "low":
                # For low risk, prioritize low volatility and beta
                if volatility > 0 and beta > 0:
                    weight = 100 / (volatility * beta)
                else:
                    weight = 10
            elif risk_preference == "high":
                # For high risk, prioritize return
                weight = max(avg_return, 5)  # Minimum weight of 5
            else:  # medium
                # For medium risk, balance return and risk
                if volatility > 0:
                    weight = (avg_return + 10) / (volatility + 10)  # Adding constants to avoid division by zero
                else:
                    weight = avg_return + 1
            
            optimized_portfolio[ticker] = weight
            total_weight += weight
        else:
            # Equal weight if data is missing
            optimized_portfolio[ticker] = 100 / len(tickers)
            total_weight += 100 / len(tickers)
    
    # Normalize to 100%
    for ticker in optimized_portfolio:
        optimized_portfolio[ticker] = round((optimized_portfolio[ticker] / total_weight) * 100, 2)
    
    return optimized_portfolio

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)