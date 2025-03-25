import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# 1. Data loading and preprocessing
def enhanced_load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Print column names and data types for debugging
    print("Columns in the dataset:", df.columns.tolist())
    print("Data types before preprocessing:")
    print(df.dtypes.head(10))
    
    # 1. Convert target column to numeric first
    target_column = 'Attrition'
    df[target_column] = df[target_column].map({'Yes': 1, 'No': 0})
    
    # 2. Feature Engineering - Create new features that capture workforce dynamics
    # Make sure to handle potential division by zero
    df['Experience_to_Promotion_Ratio'] = df['Total_Working_Years'] / (df['Years_Since_Last_Promotion'] + 1)
    df['Role_Tenure_Ratio'] = df['Years_In_Current_Role'] / (df['Years_At_Company'] + 1)
    df['Manager_Relationship_Duration'] = df['Years_With_Curr_Manager'] / (df['Years_At_Company'] + 1)
    
    # 3. Handle missing values with more sophisticated imputation
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Impute numeric columns with median (more robust than mean)
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Impute categorical columns with mode
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # 4. Encode categorical variables with more sophisticated methods
    # Use target encoding for high-cardinality features
    high_cardinality_cols = ['Job_Role', 'Department', 'Education_Field']
    
    for col in high_cardinality_cols:
        if col in df.columns:
            # Calculate mean target value for each category
            mapping = df.groupby(col)[target_column].mean().to_dict()
            # Create new encoded feature
            df[f'{col}_encoded'] = df[col].map(mapping)
            # Drop the original column to avoid string values
            df = df.drop(columns=[col])
    
    # 5. Use one-hot encoding for low-cardinality features
    low_cardinality_cols = [col for col in df.select_dtypes(include=['object']).columns]
    df = pd.get_dummies(df, columns=low_cardinality_cols, drop_first=True)
    
    # 6. Feature scaling - Standardize numeric features
    scaler = StandardScaler()
    numeric_cols_to_scale = [col for col in df.select_dtypes(include=[np.number]).columns 
                            if col != target_column]  # Don't scale the target
    df[numeric_cols_to_scale] = scaler.fit_transform(df[numeric_cols_to_scale])
    
    # 7. Check for any remaining non-numeric columns
    remaining_object_cols = df.select_dtypes(include=['object']).columns
    if len(remaining_object_cols) > 0:
        print(f"Warning: Still have object columns: {remaining_object_cols}")
        # Drop any remaining object columns
        df = df.drop(columns=remaining_object_cols)
    
    # 8. Split the dataset
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Print shape of X for debugging
    print("Shape of X:", X.shape)
    print("Number of numeric columns in X:", X.select_dtypes(include=[np.number]).shape[1])
    print("Final columns:", X.columns.tolist())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test, scaler


# 2. Load pre-trained DNN model
def load_model(model_path):
    return load_model(model_path)


# 3. Generate test sample pairs (Random Search)
def generate_sample_pair(X_test, sensitive_columns, non_sensitive_columns):
    sample_a = X_test.iloc[np.random.choice(len(X_test))]
    sample_b = sample_a.copy()

    # Apply perturbation on sensitive features (random flipping)
    for col in sensitive_columns:
        if col in X_test.columns:  # Ensure the column exists
            unique_values = X_test[col].unique()
            sample_b[col] = np.random.choice(unique_values)  # Randomly select a new value

    # Apply perturbation on non-sensitive features
    for col in non_sensitive_columns:
        if col in X_test.columns:  # Ensure the column exists
            min_val = X_test[col].min()
            max_val = X_test[col].max()
            perturbation = np.random.uniform(-0.1 * (max_val - min_val), 0.1 * (max_val - min_val))  # Small perturbation
            sample_a[col] = np.clip(sample_a[col] + perturbation, min_val, max_val)
            sample_b[col] = np.clip(sample_b[col] + perturbation, min_val, max_val)

    return sample_a, sample_b


# 4. Model prediction and individual discrimination evaluation
def evaluate_discrimination(model, sample_a, sample_b, threshold=0.05, discrimination_pairs=None):
    if discrimination_pairs is None:
        discrimination_pairs = []  # Initialize list to store discriminatory sample pairs

    # Convert sample_a and sample_b to numpy arrays and reshape
    sample_a = np.array(sample_a)
    sample_b = np.array(sample_b)

    # Model predictions
    prediction_a = model.predict(sample_a.reshape(1, -1))  # Reshape to fit model input format
    prediction_b = model.predict(sample_b.reshape(1, -1))

    # Get prediction results (usually probability values)
    pred_a = prediction_a[0][0]  # Get the value from the output (shape: (1, 1))
    pred_b = prediction_b[0][0]

    # Check if the difference in predictions is greater than the threshold
    if abs(pred_a - pred_b) > threshold:
        discrimination_pairs.append((sample_a, sample_b))  # Store the discriminatory sample pair
        return 1  # Individual discriminatory instance
    else:
        return 0  # Not a discriminatory instance


# 5. Calculate Individual Discrimination Instance Ratio (IDI ratio)
def calculate_idi_ratio(model, X_test, sensitive_columns, non_sensitive_columns, num_samples=1000):
    discrimination_count = 0

    for _ in range(num_samples):
        sample_a, sample_b = generate_sample_pair(X_test, sensitive_columns, non_sensitive_columns)
        discrimination_count += evaluate_discrimination(model, sample_a, sample_b)

    # Total number of generated samples
    total_generated = num_samples

    # Calculate Individual Discrimination Instance Ratio
    IDI_ratio = discrimination_count / total_generated
    return IDI_ratio


# 6. Main function
def main():
    # 1. Load and preprocess data with enhanced methods
    file_path = '/content/ISE-solution/lab4/dataset/HR Dataset.csv'
    X_train, X_test, y_train, y_test, scaler = enhanced_load_and_preprocess_data(file_path)
    
    # 2. Define sensitive attributes based on the paper
    # Check which columns actually exist in the processed data
    print("Available columns:", X_train.columns.tolist())
    
    # Use columns that actually exist in the processed data
    # For Gender, look for the one-hot encoded version
    gender_col = [col for col in X_train.columns if 'Gender' in col]
    age_col = ['Age'] if 'Age' in X_train.columns else []
    marital_status_cols = [col for col in X_train.columns if 'Marital_Status' in col]
    
    sensitive_columns = gender_col + age_col + marital_status_cols
    print("Using sensitive columns:", sensitive_columns)
    
    non_sensitive_columns = [col for col in X_train.columns if col not in sensitive_columns]
    
    # 3. Train a fair model
    fair_model = train_fair_model(X_train, y_train, sensitive_columns, lambda_param=0.2)
    
    # 4. Save the model
    fair_model.save('/content/ISE-solution/lab4/DNN/fair_model_hr.h5')
    
    # 5. Calculate advanced fairness metrics
    fairness_metrics = calculate_advanced_idi_ratio(
        fair_model, X_test, sensitive_columns, non_sensitive_columns)
    
    # 6. Print results
    print(f"IDI Ratio: {fairness_metrics['idi_ratio']}")
    print(f"Demographic Parity Difference: {fairness_metrics['demographic_parity_diff']}")
    print(f"Equal Opportunity Difference: {fairness_metrics['equal_opportunity_diff']}")
    
    # 7. Evaluate model performance
    y_pred = fair_model.predict(X_test).flatten() > 0.5
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_val = roc_auc_score(y_test, fair_model.predict(X_test).flatten())
    
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"AUC: {auc_val}")
    
    # 8. Compare with baseline model (optional)
    baseline_model = keras.models.load_model('/content/ISE-solution/lab4/DNN/model_processed_hr.h5')
    baseline_metrics = calculate_advanced_idi_ratio(
        baseline_model, X_test, sensitive_columns, non_sensitive_columns)
    
    print("\nComparison with baseline model:")
    print(f"Baseline IDI Ratio: {baseline_metrics['idi_ratio']}")
    print(f"Baseline Demographic Parity: {baseline_metrics['demographic_parity_diff']}")
    print(f"Baseline Equal Opportunity: {baseline_metrics['equal_opportunity_diff']}")


def fairness_regularizer(model, X, sensitive_attributes, lambda_param=0.1):
    """
    Implements a fairness regularizer based on the paper's methodology
    """
    predictions = model.predict(X)
    fairness_penalty = 0
    
    # For each sensitive attribute
    for attr in sensitive_attributes:
        if attr in X.columns:
            # Get unique values of the sensitive attribute
            unique_values = X[attr].unique()
            
            # Calculate average predictions for each group
            group_predictions = {}
            for val in unique_values:
                group_mask = X[attr] == val
                if sum(group_mask) > 0:  # Ensure group is not empty
                    group_predictions[val] = np.mean(predictions[group_mask])
            
            # Calculate variance between group predictions (measure of unfairness)
            if len(group_predictions) > 1:
                group_pred_values = list(group_predictions.values())
                fairness_penalty += np.var(group_pred_values)
    
    # Return weighted fairness penalty (default to 0 if no sensitive attributes found)
    return lambda_param * fairness_penalty


def train_fair_model(X_train, y_train, sensitive_attributes, lambda_param=0.1, epochs=50, batch_size=32):
    """
    Train a model with fairness constraints based on the paper
    """
    # Define model architecture
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Custom training loop with fairness regularization
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = keras.losses.BinaryCrossentropy()
    
    # Convert to tensor format
    X_train_tensor = tf.convert_to_tensor(X_train.values, dtype=tf.float32)
    y_train_tensor = tf.convert_to_tensor(y_train.values, dtype=tf.float32)
    
    # Training loop
    for epoch in range(epochs):
        # Shuffle the data
        indices = tf.range(start=0, limit=tf.shape(X_train_tensor)[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)
        shuffled_X = tf.gather(X_train_tensor, shuffled_indices)
        shuffled_y = tf.gather(y_train_tensor, shuffled_indices)
        
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            X_batch = shuffled_X[i:i+batch_size]
            y_batch = shuffled_y[i:i+batch_size]
            
            with tf.GradientTape() as tape:
                # Forward pass
                predictions = model(X_batch, training=True)
                # Calculate loss
                loss = loss_fn(y_batch, predictions)
                # Add fairness regularization
                fairness_loss = fairness_regularizer(model, X_train.iloc[i:i+batch_size], sensitive_attributes, lambda_param)
                total_loss = loss + fairness_loss
            
            # Backpropagation
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.numpy()}, Fairness Penalty: {fairness_loss}")
    
    return model


def calculate_advanced_idi_ratio(model, X_test, sensitive_columns, non_sensitive_columns, num_samples=1000):
    """
    Enhanced IDI ratio calculation based on the paper's methodology
    """
    discrimination_count = 0
    discrimination_pairs = []
    
    # Create a more sophisticated perturbation strategy
    for i in range(num_samples):
        # Generate counterfactual pairs with dynamic perturbation
        sample_a, sample_b = generate_dynamic_counterfactual_pair(
            X_test, sensitive_columns, non_sensitive_columns)
        
        # Evaluate discrimination with adaptive thresholding
        is_discriminatory, discrimination_score = evaluate_discrimination_advanced(
            model, sample_a, sample_b)
        
        if is_discriminatory:
            discrimination_count += 1
            discrimination_pairs.append((sample_a, sample_b, discrimination_score))
    
    # Calculate IDI ratio
    idi_ratio = discrimination_count / num_samples
    
    # Calculate additional fairness metrics from the paper
    demographic_parity_diff = calculate_demographic_parity(model, X_test, sensitive_columns)
    equal_opportunity_diff = calculate_equal_opportunity(model, X_test, y_test, sensitive_columns)
    
    return {
        'idi_ratio': idi_ratio,
        'demographic_parity_diff': demographic_parity_diff,
        'equal_opportunity_diff': equal_opportunity_diff,
        'discrimination_pairs': discrimination_pairs
    }


def generate_dynamic_counterfactual_pair(X_test, sensitive_columns, non_sensitive_columns):
    """
    Generate counterfactual pairs using dynamic perturbation strategies
    """
    # Select a random sample
    sample_idx = np.random.choice(len(X_test))
    sample_a = X_test.iloc[[sample_idx]].copy()
    sample_b = sample_a.copy()
    
    # Apply targeted perturbations to sensitive attributes
    for col in sensitive_columns:
        if col in X_test.columns:
            # Get distribution of values for this attribute
            value_counts = X_test[col].value_counts(normalize=True)
            # Select alternative value with probability inverse to its frequency
            inverse_probs = 1 / (value_counts + 0.01)  # Add small constant to avoid division by zero
            inverse_probs = inverse_probs / inverse_probs.sum()  # Normalize
            
            # Sample from inverse probability distribution
            alternative_values = value_counts.index.tolist()
            current_value = sample_a[col].values[0]
            
            # Remove current value from options
            if current_value in alternative_values:
                alternative_values.remove(current_value)
                
            if alternative_values:
                # Choose alternative value
                sample_b[col] = np.random.choice(alternative_values)
    
    # Apply adaptive perturbations to non-sensitive attributes
    for col in non_sensitive_columns:
        if col in X_test.columns and pd.api.types.is_numeric_dtype(X_test[col]):
            # Calculate perturbation based on feature importance
            # (In a real implementation, you would use feature importance scores)
            importance_factor = 0.1  # Placeholder
            
            # Calculate dynamic perturbation range
            std_dev = X_test[col].std()
            perturbation = np.random.normal(0, importance_factor * std_dev)
            
            # Apply perturbation
            sample_b[col] = sample_b[col] + perturbation
            
            # Ensure values stay within observed range
            min_val, max_val = X_test[col].min(), X_test[col].max()
            sample_b[col] = np.clip(sample_b[col], min_val, max_val)
    
    return sample_a, sample_b


def evaluate_discrimination_advanced(model, sample_a, sample_b, base_threshold=0.05):
    """
    Evaluate discrimination with adaptive thresholding
    """
    # Get model predictions
    pred_a = model.predict(sample_a).flatten()[0]
    pred_b = model.predict(sample_b).flatten()[0]
    
    # Calculate prediction difference
    pred_diff = abs(pred_a - pred_b)
    
    # Calculate confidence-based threshold
    # Higher confidence predictions should have stricter thresholds
    confidence_a = abs(pred_a - 0.5) * 2  # Scale to [0,1]
    confidence_b = abs(pred_b - 0.5) * 2
    avg_confidence = (confidence_a + confidence_b) / 2
    
    # Adaptive threshold: lower for high-confidence predictions
    adaptive_threshold = base_threshold * (1 - 0.5 * avg_confidence)
    
    # Determine if this is a discriminatory instance
    is_discriminatory = pred_diff > adaptive_threshold
    
    # Calculate discrimination score (how much it exceeds the threshold)
    discrimination_score = pred_diff / adaptive_threshold if adaptive_threshold > 0 else 0
    
    return is_discriminatory, discrimination_score


def calculate_demographic_parity(model, X, sensitive_columns):
    """
    Calculate demographic parity difference across sensitive attributes
    """
    predictions = model.predict(X).flatten() > 0.5
    
    max_diff = 0
    for col in sensitive_columns:
        if col in X.columns:
            unique_values = X[col].unique()
            group_acceptance_rates = {}
            
            for val in unique_values:
                group_mask = X[col] == val
                if sum(group_mask) > 0:
                    group_acceptance_rates[val] = np.mean(predictions[group_mask])
            
            if group_acceptance_rates:
                # Calculate max difference between any two groups
                rates = list(group_acceptance_rates.values())
                current_diff = max(rates) - min(rates)
                max_diff = max(max_diff, current_diff)
    
    return max_diff


def calculate_equal_opportunity(model, X, y, sensitive_columns):
    """
    Calculate equal opportunity difference across sensitive attributes
    """
    predictions = model.predict(X).flatten() > 0.5
    positive_mask = y == 1  # This works now that y is numeric
    
    max_diff = 0
    for col in sensitive_columns:
        if col in X.columns:
            unique_values = X[col].unique()
            group_tpr_rates = {}
            
            for val in unique_values:
                group_mask = (X[col] == val) & positive_mask
                if sum(group_mask) > 0:
                    group_tpr_rates[val] = np.mean(predictions[group_mask])
            
            if group_tpr_rates:
                # Calculate max difference between any two groups
                rates = list(group_tpr_rates.values())
                current_diff = max(rates) - min(rates) if len(rates) > 1 else 0
                max_diff = max(max_diff, current_diff)
    
    return max_diff


def fairness_performance_grid_search(X_train, y_train, X_test, y_test, sensitive_columns):
    """
    Perform grid search to find optimal fairness-performance trade-off
    """
    lambda_values = [0.05, 0.1, 0.2, 0.5, 1.0]
    results = []
    
    for lambda_param in lambda_values:
        # Train model with current lambda
        model = train_fair_model(X_train, y_train, sensitive_columns, lambda_param=lambda_param)
        
        # Calculate fairness metrics
        fairness_metrics = calculate_advanced_idi_ratio(
            model, X_test, sensitive_columns, 
            [col for col in X_test.columns if col not in sensitive_columns])
        
        # Calculate performance metrics
        y_pred = model.predict(X_test).flatten() > 0.5
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_val = roc_auc_score(y_test, model.predict(X_test).flatten())
        
        # Store results
        results.append({
            'lambda': lambda_param,
            'idi_ratio': fairness_metrics['idi_ratio'],
            'demographic_parity': fairness_metrics['demographic_parity_diff'],
            'equal_opportunity': fairness_metrics['equal_opportunity_diff'],
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc_val
        })
    
    # Find best model based on combined metric
    for result in results:
        # Calculate combined score (higher is better)
        # This formula can be adjusted based on your priorities
        result['combined_score'] = result['accuracy'] * (1 - result['idi_ratio']) * \
                                  (1 - result['demographic_parity']) * (1 - result['equal_opportunity'])
    
    # Sort by combined score (descending)
    results.sort(key=lambda x: x['combined_score'], reverse=True)
    
    return results


if __name__ == "__main__":
    main()