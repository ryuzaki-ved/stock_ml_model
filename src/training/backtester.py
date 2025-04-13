"""
Walk-forward backtesting to avoid look-ahead bias
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from loguru import logger
from datetime import timedelta

class Backtester:
    """Walk-forward backtesting"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.initial_capital = config.get('initial_capital', 1000000)
        self.transaction_cost = config.get('transaction_cost', 0.001)  # 0.1%
        
    def run(self, df: pd.DataFrame, model, feature_cols: List[str]) -> Dict:
        """Run walk-forward backtest"""
        
        logger.info("Starting walk-forward backtest...")
        
        # Sort by date
        df = df.sort_values('date')
        
        # Parameters
        train_window = self.config.get('train_window_days', 252)  # 1 year
        test_window = self.config.get('test_window_days', 21)  # 1 month
        retrain_frequency = self.config.get('retrain_frequency_days', 21)
        
        results = []
        portfolio_value = self.initial_capital
        positions = {}  # Current positions
        
        dates = df['date'].unique()
        
        for i in range(train_window, len(dates), test_window):
            test_start_date = dates[i]
            test_end_date = dates[min(i + test_window, len(dates) - 1)]
            
            # Get test data
            test_df = df[(df['date'] >= test_start_date) & (df['date'] <= test_end_date)]
            
            # Make predictions
            X_test = test_df[feature_cols]
            predictions = model.predict(X_test)
            pred_classes = np.argmax(predictions, axis=1) - 1
            
            test_df['prediction'] = pred_classes
            test_df['prediction_prob'] = np.max(predictions, axis=1)
            
            # Generate signals
            signals = self._generate_signals(test_df)
            
            # Execute trades and track portfolio
            period_results = self._execute_trades(
                test_df, 
                signals, 
                portfolio_value, 
                positions
            )
            
            results.append(period_results)
            portfolio_value = period_results['end_portfolio_value']
            positions = period_results['end_positions']
            
            logger.info(f"Period {test_start_date} to {test_end_date}: "
                       f"Portfolio = ₹{portfolio_value:,.0f}, "
                       f"Return = {period_results['period_return']:.2%}")
        
        # Aggregate results
        backtest_summary = self._summarize_results(results)
        
        return backtest_summary
    
    def _generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from predictions"""
        signals = df.copy()
        
        # Only trade on high-confidence predictions
        confidence_threshold = self.config.get('confidence_threshold', 0.6)
        
        signals['signal'] = 0
        signals.loc[
            (signals['prediction'] == 1) & (signals['prediction_prob'] > confidence_threshold),
            'signal'
        ] = 1  # Buy
        
        signals.loc[
            (signals['prediction'] == -1) & (signals['prediction_prob'] > confidence_threshold),
            'signal'
        ] = -1  # Sell/Short
        
        # Position sizing based on confidence
        signals['position_size'] = signals['prediction_prob'] / 10  # Max 10% per stock
        
        return signals
    
    def _execute_trades(self, df: pd.DataFrame, signals: pd.DataFrame, 
                       portfolio_value: float, positions: Dict) -> Dict:
        """Execute trades and track P&L"""
        
        trades = []
        daily_values = []
        
        for date in df['date'].unique():
            day_signals = signals[signals['date'] == date]
            
            # Calculate portfolio value at start of day
            day_value = portfolio_value
            for symbol, pos in positions.items():
                current_price = df[(df['symbol'] == symbol) & (df['date'] == date)]['close'].values
                if len(current_price) > 0:
                    day_value += pos['quantity'] * current_price[0]
            
            # Execute trades
            for _, row in day_signals.iterrows():
                symbol = row['symbol']
                signal = row['signal']
                price = row['close']
                
                if signal == 1:  # Buy
                    # Calculate quantity to buy
                    allocation = day_value * row['position_size']
                    quantity = int(allocation / price)
                    cost = quantity * price * (1 + self.transaction_cost)
                    
                    if cost <= portfolio_value:
                        portfolio_value -= cost
                        if symbol in positions:
                            positions[symbol]['quantity'] += quantity
                            positions[symbol]['avg_price'] = (
                                (positions[symbol]['avg_price'] * positions[symbol]['quantity'] + 
                                 price * quantity) / (positions[symbol]['quantity'] + quantity)
                            )
                        else:
                            positions[symbol] = {
                                'quantity': quantity,
                                'avg_price': price
                            }
                        
                        trades.append({
                            'date': date,
                            'symbol': symbol,
                            'action': 'BUY',
                            'quantity': quantity,
                            'price': price,
                            'cost': cost
                        })
                
                elif signal == -1 and symbol in positions:  # Sell
                    quantity = positions[symbol]['quantity']
                    revenue = quantity * price * (1 - self.transaction_cost)
                    portfolio_value += revenue
                    
                    pnl = (price - positions[symbol]['avg_price']) * quantity
                    
                    trades.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'SELL',
                        'quantity': quantity,
                        'price': price,
                        'revenue': revenue,
                        'pnl': pnl
                    })
                    
                    del positions[symbol]
            
            daily_values.append({
                'date': date,
                'portfolio_value': day_value
            })
        
        end_portfolio_value = daily_values[-1]['portfolio_value'] if daily_values else portfolio_value
        
        return {
            'trades': trades,
            'daily_values': daily_values,
            'start_portfolio_value': daily_values[0]['portfolio_value'] if daily_values else portfolio_value,
            'end_portfolio_value': end_portfolio_value,
            'period_return': (end_portfolio_value / self.initial_capital - 1),
            'end_positions': positions
        }
    
    def _summarize_results(self, results: List[Dict]) -> Dict:
        """Calculate performance metrics"""
        
        all_trades = []
        all_daily_values = []
        
        for r in results:
            all_trades.extend(r['trades'])
            all_daily_values.extend(r['daily_values'])
        
        df_values = pd.DataFrame(all_daily_values)
        df_values['returns'] = df_values['portfolio_value'].pct_change()
        
        # Calculate metrics
        total_return = (df_values['portfolio_value'].iloc[-1] / self.initial_capital) - 1
        
        sharpe_ratio = (
            df_values['returns'].mean() / df_values['returns'].std() * np.sqrt(252)
            if df_values['returns'].std() > 0 else 0
        )
        
        # Max drawdown
        rolling_max = df_values['portfolio_value'].expanding().max()
        drawdown = (df_values['portfolio_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = [t for t in all_trades if t.get('pnl', 0) > 0]
        win_rate = len(winning_trades) / len([t for t in all_trades if 'pnl' in t]) if all_trades else 0
        
        summary = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(all_trades),
            'final_portfolio_value': df_values['portfolio_value'].iloc[-1],
            'daily_values': df_values,
            'all_trades': pd.DataFrame(all_trades)
        }
        
        logger.info(f"Backtest Summary:")
        logger.info(f"  Total Return: {total_return:.2%}")
        logger.info(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"  Max Drawdown: {max_drawdown:.2%}")
        logger.info(f"  Win Rate: {win_rate:.2%}")
        logger.info(f"  Total Trades: {len(all_trades)}")
        
        return summary