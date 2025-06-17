import json
import argparse
from pathlib import Path
import tensorflow as tf
import tensorflowjs as tfjs
from models.models import * 

def load_model(model_path: str):
    model = tf.keras.models.load_model(model_path, compile=False)
    model_config = model.get_config()
    input_shape = model_config['input_shape']
    
    # Fake input        
    fake_input = tf.random.normal([1,]+input_shape)
    fake_output = model(fake_input, training=False)
    
    output_shape = fake_output.shape[1:].as_list()
    print(f"Model loaded successfully| Input shape: {input_shape} | Output shape: {output_shape}")
    
    model_config['output_shape'] = output_shape
    
    return model, model_config

def export_to_pb(model, input_shape, output_dir):
    # Create concrete function
    @tf.function
    def model_func(x):
        return model(x, training=False)
    
    concrete_func = model_func.get_concrete_function(tf.TensorSpec(shape=(None,)+tuple(input_shape), dtype=tf.float32))
    
    # Save model
    tf.saved_model.save(
        model,
        str(output_dir),
        signatures={
            "serving_default": concrete_func,
            "predict": concrete_func,
        }
    )
    
    print(f"Model exported to {output_dir}")
    
    
def export_to_tfjs(model, output_dir: str, type: str = "pb"):
    """
    type: pb, keras
    """
    if type == "pb":
        tfjs.converters.convert_tf_saved_model(
            str(model),
            str(output_dir),
        )
    elif type == "keras":
        tfjs.converters.convert_keras_model(
            model,
            str(output_dir),
        )
    else:
        raise ValueError(f"Invalid type: {type}")
    
    print(f"Model exported to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Export model")
    parser.add_argument("--model_path", type=str, default="./assets/train/best_model.keras")
    parser.add_argument("--output_dir", type=str, default="./logs/export")
    parser.add_argument("--type", type=str, default="tfjs", help="Type of export: tfjs, pb")
    args = parser.parse_args()

    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model, info = load_model(args.model_path)
    input_shape = info['input_shape']
    
    if args.type == "tfjs":
        # Convert to pb first
        pb_dir = output_dir / "pb"
        pb_dir.mkdir(parents=True, exist_ok=True)
        export_to_pb(model, input_shape, output_dir=pb_dir)
        # Convert to tfjs
        export_to_tfjs(pb_dir, output_dir / "tfjs", 'pb')
        
    elif args.type == "pb":
        # Convert to pb
        export_to_pb(model, input_shape, output_dir / 'pb')
    else:
        raise ValueError(f"Invalid type: {args.type}")
    
    # Dump infor 
    with open(output_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)

if __name__ == "__main__":
    main()
