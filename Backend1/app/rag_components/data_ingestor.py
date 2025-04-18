import yfinance as yf
from typing import List, Dict, Any
from .vector_store import VectorStoreManager

class DataIngestor:
    def __init__(self):
        """
        Initialize data ingestor with vector store manager
        """
        self.vector_store_manager = VectorStoreManager()
    
    def fetch_stock_data(self, ticker: str, period: str = "1y") -> Dict[str, Any]:
        """
        Fetch stock data for a given ticker
        
        :param ticker: Stock ticker symbol
        :param period: Data period (default: 1 year)
        :return: Dictionary of stock data
        """
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
    
    def prepare_document_text(self, stock_data: Dict[str, Any]) -> List[str]:
        """
        Prepare document text from stock data for vector store
        
        :param stock_data: Stock data dictionary
        :return: List of document texts
        """
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
        if hist_data:
            price_doc = f"""
            Historical Price Summary:
            - Average Close Price: {hist_data.get('Close', []).mean() if 'Close' in hist_data else 'N/A'}
            - Highest Price: {hist_data.get('High', []).max() if 'High' in hist_data else 'N/A'}
            - Lowest Price: {hist_data.get('Low', []).min() if 'Low' in hist_data else 'N/A'}
            """
            documents.append(price_doc)
        
        # Financial statement summary
        financials = stock_data.get('financials', {})
        for statement_type, statement_data in financials.items():
            if statement_data:
                fin_doc = f"""
                {statement_type.replace('_', ' ').title()} Summary:
                """
                for key, value in list(statement_data.items())[:5]:  # Limit to first 5 items
                    fin_doc += f"- {key}: {value}\n"
                documents.append(fin_doc)
        
        return documents
    
    def fetch_and_ingest_all_data(self, ticker: str):
        """
        Fetch stock data and ingest into vector store
        
        :param ticker: Stock ticker symbol
        """
        # Clear existing collection for this ticker
        self.vector_store_manager.clear_collection()
        
        # Fetch stock data
        stock_data = self.fetch_stock_data(ticker)
        
        # Prepare document texts
        documents = self.prepare_document_text(stock_data)
        
        # Prepare metadata
        metadata = [{"ticker": ticker, "type": "stock_info"} for _ in documents]
        
        # Add documents to vector store
        if documents:
            self.vector_store_manager.add_documents(documents, metadata)