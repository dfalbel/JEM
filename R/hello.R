
sglrd_sample <- function(model, batch_size) {

}

train <- function(model, train_dl, optimizer, loss_fn) {

  device <- model$device

  for (epoch in seq_len(config::get("n_epochs"))) {

    current_iter <- 0
    running_loss <- 0
    for (batch in enumerate(train_dl)) {

      data <- batch[[1]]$to(device = device)
      targets <- batch[[2]]$to(device = device)

      optimizer$zero_grad()
      logits <- model(data)

      loss_cl <- loss_fn(logits, targets)
      data_sample <- sglrd_sample(model)

    }
  }

}

main <- function() {

  model <- cnn(n_classes = 10)

  train_dl <- torchvision::mnist_dataset(
    root = ".",
    download = TRUE,
    transform = torchvision::transform_to_tensor
  )

  optimizer <- torch::optim_adam(
    params = model$parameters,
    lr = config::get("lr")
  )

  scheduler <- torch::lr_step(
    optimizer = optimizer,
    step_size = 1,
    gamma = config::get("decay_rate")
  )


}
