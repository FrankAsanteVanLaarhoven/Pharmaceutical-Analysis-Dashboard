from config import config
from analysis.train_pipeline import train_and_save_model
import streamlit as st

def main():
    """Run the model training pipeline"""
    print("Starting model training...")
    
    try:
        # Train and save the model
        trainer = train_and_save_model(config)
        
        if trainer is not None:
            print("\nModel training completed successfully!")
            print(f"Model saved to: {trainer.models_dir / 'grandmaster_pipeline.pkl'}")
        else:
            print("\nModel training failed!")
            
    except Exception as e:
        print(f"\nError during training: {e}")

if __name__ == "__main__":
    main() 