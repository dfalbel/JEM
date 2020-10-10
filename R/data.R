get_datasets <- function(nm) {
  
  if (nm == "mnist") {
    
    transform <- function(x) {
      torchvision::transform_to_tensor(x)$mul(2)$sub(1)
    }
    
    train_ds <- torchvision::mnist_dataset(
      root = ".",
      download = TRUE,
      transform = transform
    ) 
    
    valid_ds <- torchvision::mnist_dataset(
      root = ".", 
      train = FALSE,
      transform = transform
    )
    
    n_classes <- 10
    
  }
    
  list(train = train_ds, valid = valid_ds, n_classes = n_classes)
}