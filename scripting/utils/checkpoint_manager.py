import os
import tensorflow as tf

def create_checkpoint_manager(checkpoint_dir, model_generator, model_critic, optimizer_gen, optimizer_critic):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Define checkpoint objects
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=optimizer_gen,
        critic_optimizer=optimizer_critic,
        model_generator=model_generator,
        model_critic=model_critic
    )
    
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, directory=checkpoint_dir, max_to_keep=None
    )
    
    return checkpoint, checkpoint_manager, checkpoint_prefix

def save_checkpoint(checkpoint_manager, epoch):
    checkpoint_save_path = checkpoint_manager.save()
    print(f"Checkpoint saved at {checkpoint_save_path} for epoch {epoch + 1}")

def load_latest_checkpoint(checkpoint_manager):
    if checkpoint_manager.latest_checkpoint:
        checkpoint_manager.restore_or_initialize()
        print(f"Restored from {checkpoint_manager.latest_checkpoint}")
    else:
        print("Starting training from scratch.")