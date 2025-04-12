"""
Model training pipeline with MLflow tracking
"""
import mlflow
import mlflow.sklearn
from typing import Dict, Any
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

class ModelTrainer:
    """Train and track ML models"""
    
    def __init__(self, config: Dict):
        self.config = config
        mlflow.set_tracking_uri(config.get('mlflow_uri', 'sqlite:///mlflow.db'))
        mlflow.set_experiment(config.get('experiment_name', 'stock_prediction'))
        
    def train(self, df: pd.DataFrame, model_type: str = 'lgbm') -> Dict[str, Any]:
        """Train model with cross-validation"""
        
        # Prepare data
        feature_cols = [col for col in df.columns if col not in 
                       ['symbol', 'date', 'target', 'target_return', 'future_return']]
        
        X = df[feature_cols]
        y = df['target']
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        cv_scores = []
        
        with mlflow.start_run(run_name=f"{model_type}_training"):
            
            # Log parameters
            mlflow.log_params(self.config.get('model_params', {}))
            mlflow.log_param('n_features', len(feature_cols))
            mlflow.log_param('n_samples', len(df))
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                logger.info(f"Training fold {fold + 1}/5")
                
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train model
                if model_type == 'lgbm':
                    model = self._train_lgbm(X_train, y_train, X_val, y_val)
                
                # Evaluate
                score = self._evaluate(model, X_val, y_val)
                cv_scores.append(score)
                
                mlflow.log_metric(f'fold_{fold}_accuracy', score['accuracy'])
                mlflow.log_metric(f'fold_{fold}_sharpe', score['sharpe'])
            
            # Average CV scores
            avg_accuracy = np.mean([s['accuracy'] for s in cv_scores])
            avg_sharpe = np.mean([s['sharpe'] for s in cv_scores])
            
            mlflow.log_metric('cv_accuracy', avg_accuracy)
            mlflow.log_metric('cv_sharpe', avg_sharpe)
            
            # Train final model on all data
            final_model = self._train_lgbm(X, y)
            
            # Log model
            mlflow.sklearn.log_model(final_model, "model")
            
            # Feature importance
            self._log_feature_importance(final_model, feature_cols)
            
            logger.info(f"Training complete. CV Accuracy: {avg_accuracy:.4f}, Sharpe: {avg_sharpe:.4f}")
            
            return {
                'model': final_model,
                'cv_scores': cv_scores,
                'feature_cols': feature_cols
            }
    
    def _train_lgbm(self, X_train, y_train, X_val=None, y_val=None):
        """Train LightGBM model"""
        params = self.config.get('model_params', {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        })
        
        train_data = lgb.Dataset(X_train, label=y_train)
        
        if X_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[train_data, val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
            )
        else:
            model = lgb.train(params, train_data, num_boost_round=500)
        
        return model
    
    def _evaluate(self, model, X_val, y_val) -> Dict[str, float]:
        """Evaluate model"""
        y_pred = model.predict(X_val)
        y_pred_class = np.argmax(y_pred, axis=1) - 1  # Convert to -1, 0, 1
        
        from sklearn.metrics import accuracy_score, f1_score
        
        accuracy = accuracy_score(y_val, y_pred_class)
        f1 = f1_score(y_val, y_pred_class, average='weighted')
        
        # Calculate Sharpe ratio (assuming daily predictions)
        returns = y_pred_class * 0.01  # Simplified
        sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'sharpe': sharpe
        }
    
    def _log_feature_importance(self, model, feature_names):
        """Log feature importance plot"""
        import matplotlib.pyplot as plt
        
        importance = model.feature_importance(importance_type='gain')
        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(20)
        
        plt.figure(figsize=(10, 8))
        plt.barh(feature_imp['feature'], feature_imp['importance'])
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        
        mlflow.log_figure(plt.gcf(), "feature_importance.png")
        plt.close()