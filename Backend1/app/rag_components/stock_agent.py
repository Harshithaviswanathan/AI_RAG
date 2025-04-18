# rag_components/stock_agent.py

from langchain.chains import RetrievalQA
from langchain_community.llms import Groq
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
import yfinance as yf
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

class StockAgent:
    """Agentic interface for stock analysis using LLM and RAG"""
    def __init__(self, rag_engine, api_key):
        self.rag_engine = rag_engine
        self.api_key = api_key
        self.tools = self._create_tools()
        self.agent = self._create_agent()
    
    def _create_tools(self):
        """Create tools for the agent"""
        tools = [
            Tool(
                name="StockInfo",
                func=self._get_stock_info,
                description="Get basic information about a stock by ticker symbol"
            ),
            Tool(
                name="FundamentalAnalysis",
                func=self._get_fundamental_analysis,
                description="Get fundamental analysis of a stock including PE ratio, EPS, etc."
            ),
            Tool(
                name="TechnicalAnalysis",
                func=self._get_technical_analysis,
                description="Get technical indicators like SMA, RSI, MACD for a stock"
            ),
            Tool(
                name="RAGQuery",
                func=self._rag_query,
                description="Search for specific financial information about a company using the RAG system"
            ),
            Tool(
                name="PortfolioOptimization",
                func=self._optimize_portfolio,
                description="Optimize a portfolio based on tickers, allocations, and risk preference"
            )
        ]
        return tools
    
    def _create_agent(self):
        """Create the agent with tools"""
        llm = Groq(api_key=self.api_key, model_name="llama3-70b-8192")
        
        # Create prompt for the financial agent
        prompt = PromptTemplate.from_template(
            """
            You are a sophisticated financial analysis agent designed to help with stock analysis and portfolio optimization.
            
            Your key capabilities:
            1. Looking up current stock information
            2. Performing fundamental analysis
            3. Performing technical analysis
            4. Retrieving detailed financial information from your knowledge base
            5. Optimizing portfolios based on risk preferences
            
            To solve a problem, carefully decompose it into steps and use the appropriate tools.
            When analyzing stocks, consider both fundamental and technical factors.
            When optimizing portfolios, consider risk-reward tradeoffs.
            
            {tools}
            
            {agent_scratchpad}
            
            User's request: {input}
            """
        )
        
        agent = create_react_agent(llm, self.tools, prompt)
        
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10
        )
    
    def _get_stock_info(self, ticker: str):
        """Get basic stock information"""
        try:
            stock = yf.Ticker(ticker)
            info = {
                "name": stock.info.get("longName"),
                "symbol": stock.info.get("symbol"),
                "sector": stock.info.get("sector"),
                "industry": stock.info.get("industry"),
                "market_cap": stock.info.get("marketCap"),
                "price": stock.history(period="1d")["Close"].iloc[-1],
                "currency": stock.info.get("currency")
            }
            return json.dumps(info)
        except Exception as e:
            return f"Error retrieving stock info for {ticker}: {str(e)}"
    
    def _get_fundamental_analysis(self, ticker: str):
        """Get fundamental analysis"""
        try:
            stock = yf.Ticker(ticker)
            fundamentals = {
                "PE Ratio": stock.info.get("trailingPE"),
                "Forward PE": stock.info.get("forwardPE"),
                "PB Ratio": stock.info.get("priceToBook"),
                "EPS": stock.info.get("trailingEps"),
                "Forward EPS": stock.info.get("forwardEps"),
                "Dividend Yield": stock.info.get("dividendYield"),
                "ROE": stock.info.get("returnOnEquity"),
                "Profit Margin": stock.info.get("profitMargins"),
                "Revenue Growth": stock.info.get("revenueGrowth"),
                "Debt to Equity": stock.info.get("debtToEquity")
            }
            
            # Add analysis text
            analysis = "Fundamental Analysis:\n"
            
            # PE Ratio analysis
            pe = stock.info.get("trailingPE")
            if pe:
                if pe < 15:
                    analysis += f"- PE Ratio of {pe:.2f} is relatively low, potentially indicating undervaluation.\n"
                elif pe > 30:
                    analysis += f"- PE Ratio of {pe:.2f} is relatively high, potentially indicating overvaluation.\n"
                else:
                    analysis += f"- PE Ratio of {pe:.2f} is moderate.\n"
            
            # Dividend analysis
            div_yield = stock.info.get("dividendYield")
            if div_yield:
                if div_yield > 0.04:  # 4%
                    analysis += f"- Dividend yield of {div_yield*100:.2f}% is relatively high.\n"
                elif div_yield > 0:
                    analysis += f"- Dividend yield of {div_yield*100:.2f}%.\n"
                else:
                    analysis += "- No dividend is paid by this company.\n"
            
            # Return combined data
            return json.dumps({
                "metrics": fundamentals,
                "analysis": analysis
            })
        except Exception as e:
            return f"Error retrieving fundamental analysis for {ticker}: {str(e)}"
    
    def _get_technical_analysis(self, ticker: str):
        """Get technical indicators"""
        try:
            stock = yf.Ticker(ticker)
            history = stock.history(period="1y")
            
            if history.empty:
                return "No historical data available for this ticker."
            
            # Calculate indicators
            history["SMA_50"] = history["Close"].rolling(window=50).mean()
            history["SMA_200"] = history["Close"].rolling(window=200).mean()
            
            # RSI
            delta = history["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            history["RSI"] = 100 - (100 / (1 + rs))
            
            # MACD
            short_ema = history["Close"].ewm(span=12, adjust=False).mean()
            long_ema = history["Close"].ewm(span=26, adjust=False).mean()
            history["MACD"] = short_ema - long_ema
            history["Signal_Line"] = history["MACD"].ewm(span=9, adjust=False).mean()
            
            # Fill NaN values
            history.fillna(0, inplace=True)
            
            # Get the latest values
            latest = history.iloc[-1].to_dict()
            
            # Technical signals
            signals = {
                "Current Price": latest["Close"],
                "SMA_50": latest["SMA_50"],
                "SMA_200": latest["SMA_200"],
                "RSI": latest["RSI"],
                "MACD": latest["MACD"],
                "Signal_Line": latest["Signal_Line"],
            }
            
            # Generate analysis
            analysis = "Technical Analysis:\n"
            
            # Trend analysis based on MAs
            if latest["SMA_50"] > latest["SMA_200"]:
                analysis += "- BULLISH TREND: The 50-day moving average is above the 200-day moving average.\n"
            else:
                analysis += "- BEARISH TREND: The 50-day moving average is below the 200-day moving average.\n"
            
            # Price vs MA
            if latest["Close"] > latest["SMA_50"]:
                analysis += "- BULLISH: Price is above the 50-day moving average.\n"
            else:
                analysis += "- BEARISH: Price is below the 50-day moving average.\n"
            
            # RSI analysis
            if latest["RSI"] > 70:
                analysis += "- OVERBOUGHT: RSI is above 70, indicating the stock may be overbought.\n"
            elif latest["RSI"] < 30:
                analysis += "- OVERSOLD: RSI is below 30, indicating the stock may be oversold.\n"
            else:
                analysis += f"- NEUTRAL: RSI is at {latest['RSI']:.2f}, indicating neutral momentum.\n"
            
            # MACD analysis
            if latest["MACD"] > latest["Signal_Line"]:
                analysis += "- BULLISH SIGNAL: MACD is above the signal line, indicating bullish momentum.\n"
            else:
                analysis += "- BEARISH SIGNAL: MACD is below the signal line, indicating bearish momentum.\n"
            
            return json.dumps({
                "indicators": signals,
                "analysis": analysis
            })
        except Exception as e:
            return f"Error retrieving technical analysis for {ticker}: {str(e)}"
    
    def _rag_query(self, query_data: str):
        """Query the RAG system"""
        try:
            # Parse the query data
            try:
                data = json.loads(query_data)
                question = data.get("question", "")
                ticker = data.get("ticker", None)
            except:
                # If not valid JSON, treat the input as the question
                question = query_data
                ticker = None
                
                # Try to extract ticker from the question
                import re
                ticker_match = re.search(r'\b[A-Z]{1,5}\b', question)
                if ticker_match:
                    ticker = ticker_match.group(0)
            
            # Query the RAG system
            result = self.rag_engine.query(question, ticker)
            return json.dumps(result)
        except Exception as e:
            return f"Error querying RAG system: {str(e)}"
    
    def _optimize_portfolio(self, portfolio_request: str):
        """Optimize a portfolio"""
        try:
            # Parse the portfolio request
            data = json.loads(portfolio_request)
            tickers = data.get("tickers", [])
            risk_preference = data.get("risk_preference", "medium")
            
            if not tickers:
                return "No tickers provided for portfolio optimization."
            
            # Collect historical data
            stock_data = {}
            for ticker in tickers:
                stock = yf.Ticker(ticker)
                history = stock.history(period="1y")
                if not history.empty:
                    # Calculate returns and risk metrics
                    returns = history["Close"].pct_change().dropna()
                    stock_data[ticker] = {
                        "avg_return": returns.mean() * 252 * 100,  # Annualized return %
                        "volatility": returns.std() * np.sqrt(252) * 100,  # Annualized volatility %
                        "beta": stock.info.get("beta", 1.0),
                        "current_price": history["Close"].iloc[-1]
                    }
            
            # Simple portfolio optimization based on risk preference
            optimized_allocations = {}
            
            if risk_preference == "low":
                # For low risk: favor low volatility and beta stocks
                weights = {}
                for ticker, data in stock_data.items():
                    # Inverse of volatility * inverse of beta
                    weight = 1 / (data["volatility"] * data["beta"] if data["beta"] > 0 else 1)
                    weights[ticker] = weight
            
            elif risk_preference == "high":
                # For high risk: favor high return stocks
                weights = {}
                for ticker, data in stock_data.items():
                    # Return to volatility ratio (Sharpe-like without risk-free rate)
                    weight = max(data["avg_return"], 1) / data["volatility"] if data["volatility"] > 0 else 1
                    weights[ticker] = weight
            
            else:  # medium risk
                # For medium risk: balanced approach
                weights = {}
                for ticker, data in stock_data.items():
                    # Balance of return and risk
                    weight = (data["avg_return"] + 5) / (data["volatility"] + 5)  # Adding constants to avoid division by zero
                    weights[ticker] = weight
            
            # Normalize weights to percentages
            total_weight = sum(weights.values())
            for ticker, weight in weights.items():
                optimized_allocations[ticker] = round((weight / total_weight) * 100, 2)
            
            # Sort by allocation (highest first)
            optimized_allocations = dict(sorted(optimized_allocations.items(), key=lambda x: x[1], reverse=True))
            
            # Prepare analysis
            analysis = f"Portfolio Optimization Analysis ({risk_preference.upper()} risk profile):\n\n"
            
            # Portfolio characteristics
            total_return = sum(stock_data[ticker]["avg_return"] * (optimized_allocations[ticker]/100) for ticker in optimized_allocations)
            
            # Calculate weighted average volatility (simplified without covariance)
            portfolio_volatility = sum(stock_data[ticker]["volatility"] * (optimized_allocations[ticker]/100) for ticker in optimized_allocations)
            
            analysis += f"Expected Annual Return: {total_return:.2f}%\n"
            analysis += f"Portfolio Volatility: {portfolio_volatility:.2f}%\n\n"
            
            # Add rationale for each allocation
            analysis += "Allocation Rationale:\n"
            for ticker, allocation in optimized_allocations.items():
                if risk_preference == "low":
                    analysis += f"- {ticker}: {allocation:.2f}% - Selected for its lower volatility ({stock_data[ticker]['volatility']:.2f}%) and beta ({stock_data[ticker]['beta']:.2f}).\n"
                elif risk_preference == "high":
                    analysis += f"- {ticker}: {allocation:.2f}% - Selected for its higher potential return ({stock_data[ticker]['avg_return']:.2f}%).\n"
                else:
                    analysis += f"- {ticker}: {allocation:.2f}% - Balanced allocation based on return ({stock_data[ticker]['avg_return']:.2f}%) and risk ({stock_data[ticker]['volatility']:.2f}%).\n"
            
            return json.dumps({
                "optimized_portfolio": optimized_allocations,
                "analysis": analysis,
                "metrics": {ticker: data for ticker, data in stock_data.items()}
            })
            
        except Exception as e:
            return f"Error optimizing portfolio: {str(e)}"
    
    def run(self, query: str):
        """Run the agent with a user query"""
        return self.agent.run(query)