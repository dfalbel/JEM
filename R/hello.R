bar <- function(length) {
  progress::progress_bar$new(
    format = "[:bar] :eta Loss: :loss",
    total = length
  )
}

logsumexp <- function(x) {
  torch::torch_logsumexp(x, dim = 2)
}

sgld_sample <- function(model, x) {

  eta <- config::get("eta")
  alpha <- config::get("alpha")
  sigma <- config::get("sigma")
  
  x$requires_grad_(TRUE)
  for (i in seq_len(eta)) {
    y <- logsumexp(model(x))
    jacobian <- torch::autograd_grad(
      outputs = y,
      inputs = x,
      grad_outputs = torch::torch_ones_like(y)
    )
    noise <- torch::torch_randn(size = x$shape, device = x$device) * sigma
    x <- x + jacobian[[1]] * alpha + noise
  }
  x <- x$detach()
  
  x
}

train <- function(model, buffer, train_dl, optimizer, loss_fn) {

  device <- config::get("device")

  for (epoch in seq_len(config::get("n_epochs"))) {

    losses <- c()
    pb <- bar(length = length(train_dl))

    for (batch in torch::enumerate(train_dl)) {

      data <- batch[[1]]$to(device = device)
      targets <- batch[[2]]$to(device = device)

      # Classifier loss
      logits <- model(data)
      loss_clf <- loss_fn(logits, targets)

      # EBM loss
      # p(x)_0
      x <- buffer$get_batch(config::get("batch_size"), config::get("rho"))
      x <- sgld_sample(model, x)
      buffer$add(x)
      
      # Computes energy loss
      logsumexp_sample <- logsumexp(model(x))
      logsumexp_data <- logsumexp(model(data))
      loss_gen <- -torch::torch_mean(logsumexp_data - logsumexp_sample)

      L <- loss_clf + loss_gen

      optimizer$zero_grad()
      L$backward()
      optimizer$step()

      losses <- c(losses, L$item())
      pb$tick(tokens = list(loss = mean(losses)))

    }

    cat(sprintf("[Epoch %d] Loss: %3f\n", epoch, mean(losses)))
  }

}

main <- function() {

  model <- cnn(n_classes = 10)
  model <- mlp(n_classes = 10)
  model$to(device = config::get("device"))

  train_ds <- torchvision::mnist_dataset(
    root = ".",
    download = TRUE,
    transform = function(x) {
      x <- torchvision::transform_to_tensor(x)$mul(2)$sub(1)
      torch::torch_flatten(x, start_dim = 1)
    }
  )
  
  train_dl <- torch::dataloader(
    dataset = train_ds,
    batch_size = config::get("batch_size"),
    shuffle = TRUE
  )

  buffer <- replay_buffer(
    buffer_size = config::get("buffer_size"), 
    dim = train_ds[1][[1]]$shape
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

  loss_fn <- torch::nn_cross_entropy_loss()
  train(model, buffer, train_dl, optimizer, loss_fn)

}
