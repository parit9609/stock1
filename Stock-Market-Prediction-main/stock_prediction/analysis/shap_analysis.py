"""
SHAP Analysis Module for Model Explainability
Provides comprehensive feature importance analysis using SHAP values
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")

class SHAPAnalyzer:
    """
    SHAP-based model explainability analyzer
    
    Provides comprehensive feature importance analysis using SHAP values
    for tree-based models and other ML models.
    """
    
    def __init__(self, model, feature_names: Optional[List[str]] = None):
        """
        Initialize SHAP analyzer
        
        Args:
            model: Trained ML model
            feature_names: List of feature names
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for this analysis. Install with: pip install shap")
        
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
        # Initialize appropriate explainer based on model type
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize the appropriate SHAP explainer for the model"""
        try:
            model_type = type(self.model).__name__
            
            if hasattr(self.model, 'feature_importances_'):
                # Tree-based models (Random Forest, XGBoost, LightGBM)
                if 'XGB' in model_type or 'XGBoost' in model_type:
                    self.explainer = shap.TreeExplainer(self.model)
                elif 'LGBM' in model_type or 'LightGBM' in model_type:
                    self.explainer = shap.TreeExplainer(self.model)
                elif 'RandomForest' in model_type or 'Forest' in model_type:
                    self.explainer = shap.TreeExplainer(self.model)
                else:
                    # Generic tree explainer
                    self.explainer = shap.TreeExplainer(self.model)
                    
            elif hasattr(self.model, 'coef_'):
                # Linear models
                self.explainer = shap.LinearExplainer(self.model, self.model.scaler.transform(np.zeros((1, self.model.scaler.n_features_in_))))
                
            else:
                # Generic explainer
                self.explainer = shap.KernelExplainer(self.model.predict, np.zeros((1, 100)))
                
            logger.info(f"Initialized {type(self.explainer).__name__} for {model_type}")
            
        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {str(e)}")
            raise
    
    def explain_predictions(self, X: np.ndarray, sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate SHAP explanations for predictions
        
        Args:
            X (np.ndarray): Feature matrix
            sample_size (int): Number of samples to analyze (for large datasets)
            
        Returns:
            Dict[str, Any]: SHAP analysis results
        """
        try:
            if sample_size and len(X) > sample_size:
                # Sample data for analysis
                indices = np.random.choice(len(X), sample_size, replace=False)
                X_sample = X[indices]
                logger.info(f"Sampling {sample_size} instances from {len(X)} total")
            else:
                X_sample = X
            
            # Generate SHAP values
            if hasattr(self.model, 'coef_'):
                # For linear models, use the explainer directly
                self.shap_values = self.explainer.shap_values(X_sample)
            else:
                # For tree-based models
                self.shap_values = self.explainer.shap_values(X_sample)
            
            # Ensure shap_values is a list/array
            if isinstance(self.shap_values, list):
                self.shap_values = self.shap_values[0]  # Take first element for regression
            
            logger.info(f"Generated SHAP values for {len(X_sample)} samples")
            
            return {
                'shap_values': self.shap_values,
                'feature_names': self.feature_names,
                'X_sample': X_sample,
                'sample_size': len(X_sample)
            }
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {str(e)}")
            raise
    
    def get_feature_importance(self, X: np.ndarray, method: str = 'mean_abs') -> pd.DataFrame:
        """
        Calculate feature importance based on SHAP values
        
        Args:
            X (np.ndarray): Feature matrix
            method (str): Method for importance calculation ('mean_abs', 'mean', 'max')
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        try:
            # Generate SHAP explanations if not already done
            if self.shap_values is None:
                self.explain_predictions(X)
            
            # Calculate feature importance
            if method == 'mean_abs':
                importance = np.mean(np.abs(self.shap_values), axis=0)
            elif method == 'mean':
                importance = np.mean(self.shap_values, axis=0)
            elif method == 'max':
                importance = np.max(np.abs(self.shap_values), axis=0)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'Feature': self.feature_names or [f'Feature_{i}' for i in range(len(importance))],
                'Importance': importance,
                'Abs_Importance': np.abs(importance)
            })
            
            # Sort by absolute importance
            importance_df = importance_df.sort_values('Abs_Importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            raise
    
    def plot_feature_importance(self, X: np.ndarray, top_n: int = 20, 
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance based on SHAP values
        
        Args:
            X (np.ndarray): Feature matrix
            top_n (int): Number of top features to show
            save_path (str): Path to save the plot
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        try:
            importance_df = self.get_feature_importance(X)
            
            # Select top features
            top_features = importance_df.head(top_n)
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
            
            # Bar plot of absolute importance
            ax1.barh(range(len(top_features)), top_features['Abs_Importance'])
            ax1.set_yticks(range(len(top_features)))
            ax1.set_yticklabels(top_features['Feature'])
            ax1.set_xlabel('Mean |SHAP Value|')
            ax1.set_title(f'Top {top_n} Feature Importance (SHAP)')
            ax1.grid(True, alpha=0.3)
            
            # Horizontal bar plot for better readability
            ax2.barh(range(len(top_features)), top_features['Importance'])
            ax2.set_yticks(range(len(top_features)))
            ax2.set_yticklabels(top_features['Feature'])
            ax2.set_xlabel('Mean SHAP Value')
            ax2.set_title(f'Top {top_n} Feature Impact (SHAP)')
            ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Feature importance plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")
            raise
    
    def plot_summary_plot(self, X: np.ndarray, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create SHAP summary plot
        
        Args:
            X (np.ndarray): Feature matrix
            save_path (str): Path to save the plot
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        try:
            # Generate SHAP explanations if not already done
            if self.shap_values is None:
                self.explain_predictions(X)
            
            # Create summary plot
            fig = plt.figure(figsize=(12, 8))
            shap.summary_plot(
                self.shap_values, 
                X, 
                feature_names=self.feature_names,
                show=False
            )
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP summary plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating summary plot: {str(e)}")
            raise
    
    def plot_dependence_plot(self, X: np.ndarray, feature_idx: int, 
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Create SHAP dependence plot for a specific feature
        
        Args:
            X (np.ndarray): Feature matrix
            feature_idx (int): Index of feature to analyze
            save_path (str): Path to save the plot
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        try:
            # Generate SHAP explanations if not already done
            if self.shap_values is None:
                self.explain_predictions(X)
            
            # Create dependence plot
            fig = plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                feature_idx,
                self.shap_values,
                X,
                feature_names=self.feature_names,
                show=False
            )
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP dependence plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating dependence plot: {str(e)}")
            raise
    
    def plot_waterfall_plot(self, X: np.ndarray, sample_idx: int = 0,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Create SHAP waterfall plot for a specific sample
        
        Args:
            X (np.ndarray): Feature matrix
            sample_idx (int): Index of sample to analyze
            save_path (str): Path to save the plot
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        try:
            # Generate SHAP explanations if not already done
            if self.shap_values is None:
                self.explain_predictions(X)
            
            # Create waterfall plot
            fig = plt.figure(figsize=(12, 8))
            shap.waterfall_plot(
                shap.Explanation(
                    values=self.shap_values[sample_idx],
                    base_values=self.explainer.expected_value,
                    data=X[sample_idx],
                    feature_names=self.feature_names
                ),
                show=False
            )
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP waterfall plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating waterfall plot: {str(e)}")
            raise
    
    def create_comprehensive_analysis(self, X: np.ndarray, output_dir: str,
                                    sample_size: int = 1000) -> Dict[str, Any]:
        """
        Create comprehensive SHAP analysis with all plots
        
        Args:
            X (np.ndarray): Feature matrix
            output_dir (str): Directory to save analysis results
            sample_size (int): Number of samples to analyze
            
        Returns:
            Dict[str, Any]: Analysis results summary
        """
        try:
            from pathlib import Path
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("Starting comprehensive SHAP analysis...")
            
            # Generate SHAP explanations
            analysis_results = self.explain_predictions(X, sample_size)
            
            # Get feature importance
            importance_df = self.get_feature_importance(X)
            importance_df.to_csv(output_dir / 'feature_importance_shap.csv', index=False)
            
            # Create plots
            plots_created = []
            
            # 1. Feature importance plot
            try:
                fig = self.plot_feature_importance(X, save_path=output_dir / 'feature_importance_shap.png')
                plots_created.append('feature_importance')
                plt.close(fig)
            except Exception as e:
                logger.warning(f"Could not create feature importance plot: {e}")
            
            # 2. Summary plot
            try:
                fig = self.plot_summary_plot(X, save_path=output_dir / 'shap_summary.png')
                plots_created.append('summary')
                plt.close(fig)
            except Exception as e:
                logger.warning(f"Could not create summary plot: {e}")
            
            # 3. Dependence plots for top features
            try:
                top_features = importance_df.head(5)
                for i, (_, row) in enumerate(top_features.iterrows()):
                    feature_name = row['Feature']
                    # Find feature index
                    if self.feature_names:
                        feature_idx = self.feature_names.index(feature_name)
                    else:
                        feature_idx = i
                    
                    fig = self.plot_dependence_plot(
                        X, feature_idx, 
                        save_path=output_dir / f'dependence_{feature_name}.png'
                    )
                    plots_created.append(f'dependence_{feature_name}')
                    plt.close(fig)
            except Exception as e:
                logger.warning(f"Could not create dependence plots: {e}")
            
            # 4. Waterfall plot for sample
            try:
                fig = self.plot_waterfall_plot(X, save_path=output_dir / 'waterfall_sample.png')
                plots_created.append('waterfall')
                plt.close(fig)
            except Exception as e:
                logger.warning(f"Could not create waterfall plot: {e}")
            
            # Create analysis summary
            summary = {
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'sample_size_analyzed': analysis_results['sample_size'],
                'total_features': len(self.feature_names) if self.feature_names else X.shape[1],
                'plots_created': plots_created,
                'top_features': importance_df.head(10).to_dict('records'),
                'shap_values_shape': self.shap_values.shape if self.shap_values is not None else None
            }
            
            # Save summary
            import json
            with open(output_dir / 'shap_analysis_summary.json', 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Comprehensive SHAP analysis completed. Results saved to {output_dir}")
            return summary
            
        except Exception as e:
            logger.error(f"Error in comprehensive SHAP analysis: {str(e)}")
            raise

def analyze_model_explainability(model, X: np.ndarray, feature_names: Optional[List[str]] = None,
                               output_dir: str = "output/shap_analysis") -> Dict[str, Any]:
    """
    Convenience function to analyze model explainability
    
    Args:
        model: Trained ML model
        X (np.ndarray): Feature matrix
        feature_names (List[str]): List of feature names
        output_dir (str): Directory to save analysis results
        
    Returns:
        Dict[str, Any]: Analysis results
    """
    try:
        analyzer = SHAPAnalyzer(model, feature_names)
        return analyzer.create_comprehensive_analysis(X, output_dir)
    except Exception as e:
        logger.error(f"Error analyzing model explainability: {str(e)}")
        raise
