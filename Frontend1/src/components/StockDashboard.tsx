// StockDashboard.tsx
import React, { useState, useRef, useEffect, useContext } from "react";
import axios from "axios";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Table, TableHead, TableRow, TableHeader, TableBody, TableCell } from "@/components/ui/table";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Loader2, Send, Database, FileText } from "lucide-react";

// Create simple Tabs components to avoid dependency issues
// Custom Tabs component implementation without cloneElement
const Tabs: React.FC<{
  defaultValue: string;
  className?: string;
  children: React.ReactNode;
}> = ({ defaultValue, className, children }) => {
  const [value, setValue] = useState(defaultValue);
  
  // Create a context to share the active tab value with children
  const tabsContext = {
    value,
    setValue
  };
  
  return (
    <TabsContext.Provider value={tabsContext}>
      <div className={className}>
        {children}
      </div>
    </TabsContext.Provider>
  );
};

// Create a context to share tab state
const TabsContext = React.createContext<{
  value: string;
  setValue: React.Dispatch<React.SetStateAction<string>>;
}>({
  value: '',
  setValue: () => {}
});

const TabsList: React.FC<{
  className?: string;
  children: React.ReactNode;
}> = ({ className, children }) => {
  return (
    <div className={`inline-flex h-10 items-center justify-center rounded-md bg-muted p-1 text-muted-foreground ${className || ''}`}>
      {children}
    </div>
  );
};

const TabsTrigger: React.FC<{
  value: string;
  className?: string;
  children: React.ReactNode;
}> = ({ value, className, children }) => {
  const { value: activeTab, setValue } = useContext(TabsContext);
  
  return (
    <button
      className={`inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 ${
        activeTab === value 
          ? 'bg-background text-foreground shadow-sm' 
          : ''
      } ${className || ''}`}
      onClick={() => setValue(value)}
    >
      {children}
    </button>
  );
};

const TabsContent: React.FC<{
  value: string;
  className?: string;
  children: React.ReactNode;
}> = ({ value, className, children }) => {
  const { value: activeTab } = useContext(TabsContext);
  
  if (activeTab !== value) return null;
  
  return (
    <div className={`mt-2 ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 ${className || ''}`}>
      {children}
    </div>
  );
};
// API base URL
const API_BASE_URL = "http://localhost:8000";

// Define types for the application
interface Message {
  role: "user" | "assistant";
  content: string;
}

interface IngestionStatus {
  loading: boolean;
  success: boolean | null;
}

interface StockInfo {
  name: string;
  symbol: string;
  sector: string;
  industry: string;
  market_cap: number;
  price: number;
  currency: string;
}

interface FundamentalAnalysis {
  metrics: Record<string, number | string | null>;
  analysis: string;
}

interface TechnicalAnalysis {
  indicators: Record<string, number | string | null>;
  analysis: string;
}

interface RAGInsights {
  answer: string;
  sources: Array<{
    source: string;
    ticker?: string;
    publish_date?: number;
    [key: string]: any;
  }>;
}

interface ComprehensiveAnalysisData {
  stock_info: StockInfo;
  fundamental_analysis: FundamentalAnalysis;
  technical_analysis: TechnicalAnalysis;
  rag_insights: RAGInsights;
}

interface PortfolioStock {
  ticker: string;
  allocation: number;
}

// StockAgentChat component for interacting with the RAG agent
const StockAgentChat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", content: "Hello! I'm your RAG-powered stock analysis agent. How can I help you today?" }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [ticker, setTicker] = useState("");
  const [ingestionStatus, setIngestionStatus] = useState<IngestionStatus>({ loading: false, success: null });
  const messageEndRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Handle sending a message to the agent
  const handleSendMessage = async () => {
    if (!input.trim()) return;

    // Add user message to chat
    const userMessage: Message = { role: "user", content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/rag/agent_query`, {
        query: input
      });

      // Add assistant response to chat
      setMessages(prev => [
        ...prev,
        { role: "assistant", content: response.data.result}
      ]);
    } catch (error) {
      console.error("Error querying agent:", error);
      setMessages(prev => [
        ...prev,
        { role: "assistant", content: "Sorry, I encountered an error processing your request. Please try again." }
      ]);
    } finally {
      setLoading(false);
    }
  };

  // Handle RAG query for a specific question
  const handleRAGQuery = async () => {
    if (!input.trim()) return;

    // Add user message to chat
    const userMessage: Message = { role: "user", content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/rag/query`, {
        question: input,
        ticker: ticker.trim() || null
      });

      // Format sources for display
      const sources = response.data.sources.map((source: any) => 
        `[${source.source}${source.ticker ? ` - ${source.ticker}` : ''}]`
      ).join(", ");

      // Add assistant response to chat
      setMessages(prev => [
        ...prev,
        { 
          role: "assistant", 
          content: `${response.data.answer}\n\nSources: ${sources}`
        }
      ]);
    } catch (error) {
      console.error("Error querying RAG system:", error);
      setMessages(prev => [
        ...prev,
        { role: "assistant", content: "Sorry, I encountered an error processing your request. Please try again." }
      ]);
    } finally {
      setLoading(false);
    }
  };

  // Handle data ingestion for a ticker
  const handleIngestData = async () => {
    if (!ticker.trim()) return;
    
    setIngestionStatus({ loading: true, success: null });
    
    try {
      await axios.post(`${API_BASE_URL}/rag/ingest_stock_data`, {
        ticker: ticker.trim()
      });
      
      setIngestionStatus({ loading: false, success: true });
      
      // Add system message to chat
      setMessages(prev => [
        ...prev,
        { role: "assistant", content: `✅ Successfully ingested new data for ${ticker.trim()}. You can now ask detailed questions about this stock.` }
      ]);
    } catch (error) {
      console.error("Error ingesting data:", error);
      setIngestionStatus({ loading: false, success: false });
      
      // Add error message to chat
      setMessages(prev => [
        ...prev,
        { role: "assistant", content: `❌ Failed to ingest data for ${ticker.trim()}. Please try again or check if the ticker is valid.` }
      ]);
    }
  };

  return (
    <Card className="w-full max-w-4xl">
      <CardHeader>
        <CardTitle>RAG Stock Analysis Agent</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex gap-2 mb-4">
          <Input
            placeholder="Enter stock ticker (e.g., AAPL)"
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            className="w-48"
          />
          <Button 
            onClick={handleIngestData}
            disabled={ingestionStatus.loading || !ticker.trim()}
            className="flex gap-2 items-center"
          >
            {ingestionStatus.loading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" /> Loading
              </>
            ) : (
              <>
                <Database className="h-4 w-4" /> Ingest Data
              </>
            )}
          </Button>
          
          {ingestionStatus.success !== null && (
            <Alert variant={ingestionStatus.success ? "default" : "destructive"} className="ml-2">
              <AlertDescription>
                {ingestionStatus.success 
                  ? `Data for ${ticker} successfully ingested.` 
                  : `Failed to ingest data for ${ticker}.`}
              </AlertDescription>
            </Alert>
          )}
        </div>
        
        <Tabs defaultValue="agent" className="mb-4">
          <TabsList className="grid grid-cols-2">
            <TabsTrigger value="agent">Agent Chat</TabsTrigger>
            <TabsTrigger value="rag">RAG Query</TabsTrigger>
          </TabsList>
          
          <TabsContent value="agent" className="mt-4">
            <div className="text-sm text-muted-foreground mb-2">
              Ask the agent any stock-related question or request portfolio optimization
            </div>
            <div className="flex gap-2">
              <Input
                placeholder="Ask me anything about stocks or portfolios..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                className="flex-1"
              />
              <Button 
                onClick={handleSendMessage}
                disabled={loading || !input.trim()}
                className="flex gap-2 items-center"
              >
                {loading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
            </div>
          </TabsContent>
          
          <TabsContent value="rag" className="mt-4">
            <div className="text-sm text-muted-foreground mb-2">
              Search the knowledge base for specific information (more accurate for factual queries)
            </div>
            <div className="flex gap-2">
              <Input
                placeholder="Search for specific information..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleRAGQuery()}
                className="flex-1"
              />
              <Button 
                onClick={handleRAGQuery}
                disabled={loading || !input.trim()}
                className="flex gap-2 items-center"
              >
                {loading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <FileText className="h-4 w-4" />
                )}
              </Button>
            </div>
          </TabsContent>
        </Tabs>
        
        <div className="bg-secondary p-4 rounded-md h-96 overflow-y-auto mb-4">
          {messages.map((message, index) => (
            <div key={index} className={`mb-4 ${message.role === 'user' ? 'text-right' : 'text-left'}`}>
              <div className={`inline-block p-3 rounded-lg ${
                message.role === 'user' 
                  ? 'bg-primary text-primary-foreground' 
                  : 'bg-muted'
              }`}>
                {message.content.split('\n').map((line, i) => (
                  <React.Fragment key={i}>
                    {line}
                    {i < message.content.split('\n').length - 1 && <br />}
                  </React.Fragment>
                ))}
              </div>
            </div>
          ))}
          <div ref={messageEndRef} />
        </div>
      </CardContent>
    </Card>
  );
};

// ComprehensiveAnalysis component for detailed stock analysis
const ComprehensiveAnalysis: React.FC = () => {
  const [ticker, setTicker] = useState("");
  const [analysis, setAnalysis] = useState<ComprehensiveAnalysisData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchAnalysis = async () => {
    if (!ticker.trim()) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.get<ComprehensiveAnalysisData>(`${API_BASE_URL}/rag/comprehensive_analysis/${ticker.trim()}`);
      setAnalysis(response.data);
    } catch (error) {
      console.error("Error fetching analysis:", error);
      setError("Failed to fetch comprehensive analysis. Please try again or check if the ticker is valid.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="w-full max-w-4xl">
      <CardHeader>
        <CardTitle>Comprehensive Stock Analysis</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex gap-2 mb-4">
          <Input
            placeholder="Enter stock ticker (e.g., AAPL)"
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            className="w-64"
          />
          <Button 
            onClick={fetchAnalysis}
            disabled={loading || !ticker.trim()}
            className="flex gap-2 items-center"
          >
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" /> Analyzing
              </>
            ) : (
              "Analyze"
            )}
          </Button>
        </div>
        
        {error && (
          <Alert variant="destructive" className="mb-4">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
        
        {analysis && (
          <div className="space-y-4">
            {/* Stock Basic Info */}
            <Card>
              <CardHeader className="py-2">
                <CardTitle className="text-lg">{analysis.stock_info.name} ({analysis.stock_info.symbol})</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <p className="text-sm text-muted-foreground">Sector</p>
                    <p>{analysis.stock_info.sector || "N/A"}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Current Price</p>
                    <p>{analysis.stock_info.price?.toFixed(2) || "N/A"}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Market Cap</p>
                    <p>{analysis.stock_info.market_cap ? (analysis.stock_info.market_cap / 1e9).toFixed(2) + "B" : "N/A"}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            {/* Tabs for different analyses */}
            <Tabs defaultValue="fundamental">
              <TabsList className="grid grid-cols-3">
                <TabsTrigger value="fundamental">Fundamental</TabsTrigger>
                <TabsTrigger value="technical">Technical</TabsTrigger>
                <TabsTrigger value="rag">RAG Insights</TabsTrigger>
              </TabsList>
              
              {/* Fundamental Analysis Tab */}
              <TabsContent value="fundamental" className="mt-4">
                <Card>
                  <CardHeader className="py-2">
                    <CardTitle className="text-lg">Fundamental Analysis</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableHeader>Metric</TableHeader>
                          <TableHeader>Value</TableHeader>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {Object.entries(analysis.fundamental_analysis.metrics).map(([key, value]) => (
                          <TableRow key={key}>
                            <TableCell>{key}</TableCell>
                            <TableCell>{value !== null ? (typeof value === 'number' ? value.toFixed(2) : String(value)) : 'N/A'}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                    
                    <div className="mt-4">
                      <h4 className="font-medium mb-2">Analysis</h4>
                      <p className="whitespace-pre-line">{analysis.fundamental_analysis.analysis}</p>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
              
              {/* Technical Analysis Tab */}
              <TabsContent value="technical" className="mt-4">
                <Card>
                  <CardHeader className="py-2">
                    <CardTitle className="text-lg">Technical Analysis</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableHeader>Indicator</TableHeader>
                          <TableHeader>Value</TableHeader>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {Object.entries(analysis.technical_analysis.indicators).map(([key, value]) => (
                          <TableRow key={key}>
                            <TableCell>{key}</TableCell>
                            <TableCell>{value !== null ? (typeof value === 'number' ? value.toFixed(2) : String(value)) : 'N/A'}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                    
                    <div className="mt-4">
                      <h4 className="font-medium mb-2">Analysis</h4>
                      <p className="whitespace-pre-line">{analysis.technical_analysis.analysis}</p>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
              
              {/* RAG Insights Tab */}
              <TabsContent value="rag" className="mt-4">
                <Card>
                  <CardHeader className="py-2">
                    <CardTitle className="text-lg">RAG-Generated Insights</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="prose max-w-none">
                      <p className="whitespace-pre-line">{analysis.rag_insights.answer}</p>
                      
                      <h4 className="text-sm font-medium mt-4">Sources</h4>
                      <ul className="text-sm">
                        {analysis.rag_insights.sources.map((source, index) => (
                          <li key={index}>
                            {source.source} 
                            {source.ticker && ` - ${source.ticker}`}
                            {source.publish_date && ` (${new Date(source.publish_date * 1000).toLocaleDateString()})`}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

// PortfolioOptimizerRAG component for RAG-enhanced portfolio optimization
const PortfolioOptimizerRAG: React.FC = () => {
  const [portfolio, setPortfolio] = useState<PortfolioStock[]>([
    { ticker: "AAPL", allocation: 25 },
    { ticker: "MSFT", allocation: 25 },
    { ticker: "GOOGL", allocation: 25 },
    { ticker: "AMZN", allocation: 25 }
  ]);
  const [riskPreference, setRiskPreference] = useState<string>("medium");
  const [optimizedPortfolio, setOptimizedPortfolio] = useState<Record<string, number> | null>(null);
  const [analysis, setAnalysis] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const addStock = () => {
    setPortfolio([...portfolio, { ticker: "", allocation: 0 }]);
  };

  const removeStock = (index: number) => {
    const newPortfolio = [...portfolio];
    newPortfolio.splice(index, 1);
    setPortfolio(newPortfolio);
  };

  const updatePortfolio = (index: number, field: string, value: any) => {
    const newPortfolio = [...portfolio];
    if (field === "ticker") {
      newPortfolio[index].ticker = value.toUpperCase();
    } else if (field === "allocation") {
      newPortfolio[index].allocation = parseFloat(value);
    }
    setPortfolio(newPortfolio);
  };

  const optimizePortfolio = async () => {
    // Validate portfolio
    const emptyTickers = portfolio.some(stock => !stock.ticker.trim());
    if (emptyTickers) {
      setError("Please fill in all ticker symbols");
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      // Convert portfolio to the format expected by the API
      const tickers = portfolio.map(stock => stock.ticker);
      const allocations = portfolio.reduce<Record<string, number>>((acc, stock) => {
        acc[stock.ticker] = stock.allocation;
        return acc;
      }, {});
      
      const response = await axios.post(`${API_BASE_URL}/rag/optimize_portfolio`, {
        tickers,
        allocations,
        risk_preference: riskPreference
      });
      
      setOptimizedPortfolio(response.data.optimized_portfolio);
      setAnalysis(response.data.analysis);
    } catch (error) {
      console.error("Error optimizing portfolio:", error);
      setError("Failed to optimize portfolio. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="w-full max-w-4xl">
      <CardHeader>
        <CardTitle>RAG-Enhanced Portfolio Optimizer</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="mb-4">
          <h3 className="text-lg font-medium mb-2">Your Portfolio</h3>
          
          {portfolio.map((stock, index) => (
            <div key={index} className="flex items-center gap-2 mb-2">
              <Input
                placeholder="Stock ticker"
                value={stock.ticker}
                onChange={(e) => updatePortfolio(index, "ticker", e.target.value)}
                className="w-32"
              />
              <Input
                type="number"
                placeholder="Allocation %"
                value={stock.allocation}
                onChange={(e) => updatePortfolio(index, "allocation", e.target.value)}
                className="w-32"
              />
              <Button variant="outline" size="sm" onClick={() => removeStock(index)} disabled={portfolio.length <= 1}>
                Remove
              </Button>
            </div>
          ))}
          
          <Button variant="outline" onClick={addStock} className="mt-2">
            Add Stock
          </Button>
        </div>
        
        <div className="mb-4">
          <h3 className="text-lg font-medium mb-2">Risk Preference</h3>
          <div className="flex gap-2">
            {["low", "medium", "high"].map((risk) => (
              <Button
                key={risk}
                variant={riskPreference === risk ? "default" : "outline"}
                onClick={() => setRiskPreference(risk)}
                className="capitalize"
              >
                {risk}
              </Button>
            ))}
          </div>
        </div>
        
        {error && (
          <Alert variant="destructive" className="mb-4">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
        
        <Button 
          onClick={optimizePortfolio}
          disabled={loading}
          className="flex gap-2 items-center w-full"
        >
          {loading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" /> Optimizing
            </>
          ) : (
            "Optimize Portfolio"
          )}
        </Button>
        
        {optimizedPortfolio && (
          <div className="mt-4 space-y-4">
            <Card>
              <CardHeader className="py-2">
                <CardTitle className="text-lg">Optimized Portfolio</CardTitle>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableHeader>Stock</TableHeader>
                      <TableHeader>Allocation (%)</TableHeader>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {Object.entries(optimizedPortfolio).map(([ticker, allocation]) => (
                      <TableRow key={ticker}>
                        <TableCell>{ticker}</TableCell>
                        <TableCell>{typeof allocation === 'number' ? allocation.toFixed(2) : String(allocation)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
            
            {analysis && (
              <Card>
                <CardHeader className="py-2">
                  <CardTitle className="text-lg">Analysis</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="whitespace-pre-line">{analysis}</p>
                </CardContent>
              </Card>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

// Main StockAnalysisRAG component that integrates all RAG components
const StockAnalysisRAG: React.FC = () => {
  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-6">RAG-Enhanced Stock Analysis Platform</h1>
      
      <Tabs defaultValue="agent" className="mb-8">
        <TabsList className="grid grid-cols-3">
          <TabsTrigger value="agent">Agent Chat</TabsTrigger>
          <TabsTrigger value="analysis">Stock Analysis</TabsTrigger>
          <TabsTrigger value="portfolio">Portfolio Optimizer</TabsTrigger>
        </TabsList>
        
        <TabsContent value="agent" className="mt-4">
          <StockAgentChat />
        </TabsContent>
        
        <TabsContent value="analysis" className="mt-4">
          <ComprehensiveAnalysis />
        </TabsContent>
        
        <TabsContent value="portfolio" className="mt-4">
          <PortfolioOptimizerRAG />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default StockAnalysisRAG;