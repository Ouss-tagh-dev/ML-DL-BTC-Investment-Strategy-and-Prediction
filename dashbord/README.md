# Bitcoin ML/DL Dashboard

A state-of-the-art React dashboard for visualizing Bitcoin algorithmic trading strategies and real-time AI consensus.

## 🌟 Features

- **Real-Time AI Consensus**: Visualizes the aggregate decision of 9 ML/DL models (Buy/Sell/Neutral) with detailed vote breakdown.
- **Inference Hub**: Live control panel to trigger real-time predictions on the latest market data.
- **Dynamic UI**: Glassmorphic design with animated backgrounds and context-aware status indicators.
- **Performance Analytics**: Interactive charts comparing model accuracy, Sharpe ratios, and cumulative returns.
- **Live Status Monitoring**: Real-time "API Live Hub" indicator for backend connectivity.

## 🛠️ Tech Stack

- **Frontend**: React 18, React Router v6
- **Styling**: Tailwind CSS, Lucide React (Icons)
- **State Management**: TanStack Query (React Query)
- **Charts**: Recharts, Lightweight Charts

## 📦 Installation

1.  **Navigate to the dashboard directory**:
    ```bash
    cd dashbord
    ```
2.  **Install dependencies**:
    ```bash
    npm install
    ```

## 🚀 Usage

### Start the Development Server

```bash
npm start
```
*The dashboard will launch at `http://localhost:3000`.*

### Configuration

The dashboard connects to the backend at `http://localhost:8000`. Ensure the backend server is running for full functionality.

## 🖥️ Key Views

- **Overview**: Central command center with AI Consensus, live price, and key performance metrics.
- **Models**: Detailed cards for each of the 9 models, allowing individual inspection and manual inference.
- **Analytics**: Comparative analysis of all models (Accuracy vs. Risk).
- **Inference**: Dedicated interface for running batch predictions and viewing feature engineering details.

## 🎨 Design

The interface features a custom "Cyber-Glass" aesthetic:
- **Alpha Overlay**: CRT-style scanlines for a technical feel.
- **Mesh Float**: Subtle animated background mesh.
- **Interactive Elements**: Hover effects, pulse animations for live status, and smooth transitions.
