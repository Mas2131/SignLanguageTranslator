from Dataset import load_dataset


#Prints for debugging
def main():
    print("Starting dataset loading...")
    try:
        train_data, test_data, train_label, test_label = load_dataset()
        print("Dataset loaded successfully!")
        print(f"Train data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        print(f"Train labels shape: {train_label.shape}")
        print(f"Test labels shape: {test_label.shape}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the main function
if __name__ == "__main__":
    main()