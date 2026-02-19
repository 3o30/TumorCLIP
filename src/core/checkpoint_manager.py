import os
import json
import numpy as np
from datetime import datetime


class CheckpointManager:

    def __init__(self, checkpoint_path="experiment_checkpoint.json"):
        self.checkpoint_path = checkpoint_path

    def convert_numpy_types(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        else:
            return obj

    def save_checkpoint(self, results, current_model_index, current_trial_index,
                       current_model_partial_results=None):

        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'current_model_index': current_model_index,
            'current_trial_index': current_trial_index,
            'completed_results': results,
            'current_model_partial_results': current_model_partial_results or {}
        }

        # Ensure JSON serialization compatibility by converting NumPy types
        converted_data = self.convert_numpy_types(checkpoint_data)

        with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)

        print(f"Checkpoint saved: Model {current_model_index+1}, Trial {current_trial_index+1}")

    def load_checkpoint(self):
        
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)

                print(f"Checkpoint found: {checkpoint_data['timestamp']}")
                print(f"Completed models: {len(checkpoint_data['completed_results'])}")

                return checkpoint_data

            except Exception as e:
                print(f"Checkpoint file corrupted: {e}")
                return None

        return None

    def clear_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
            print("Checkpoint file removed")
