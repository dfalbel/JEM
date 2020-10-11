
logsumexp <- function(x) {
  torch::torch_logsumexp(x, dim = 2)
}

sgld_sample <- function(model, x, config = config::get()) {

  eta <- config$eta
  alpha <- config$alpha
  sigma <- config$sigma
  
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

train <- function(model, buffer, train_dl, optimizer, loss_fn, config = config::get()) {

  device <- config$device

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
    x <- buffer$get_batch(config$batch_size, config$rho)
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
  
  losses
}

valid <- function(model, valid_dl, loss_fn, config = config::get()) {
  
  model$eval()
  correct <- 0
  n <- 0
  losses <- c()
  
  torch::with_no_grad({
    for (batch in torch::enumerate(valid_dl)) {
      
      data <- batch[[1]]$to(device = config$device)
      targets <- batch[[2]]$to(device = config$device)
      
      logits <- model(data)
      predicted <- torch::torch_max(logits, dim = 2)[[2]]
      
      correct <- correct + predicted$eq(targets)$sum()$item()
      
      loss <- loss_fn(logits, targets)
      losses <- c(losses, loss$item())
     
      n <- n + targets$shape[1] 
    }
  })
  
  acc <- correct/n
  list(acc = acc, correct = correct, n = n, loss = mean(losses))
} 


run_experiment <- function(config = config::get()) {

  ds <- get_datasets(config$dataset)
  
  train_dl <- torch::dataloader(
    dataset = ds$train,
    batch_size = config$batch_size,
    shuffle = TRUE
  )
  
  valid_dl <- torch::dataloader(
    dataset = ds$valid,
    batch_size = config$batch_size,
    shuffle = FALSE
  )

  buffer <- replay_buffer(
    buffer_size = config$batch_size, 
    dim = ds$train[1][[1]]$shape
  )
  
  if (config$model == "cnn")
    model <- cnn(n_classes = ds$n_classes)
  else if (config$model == "mlp")
    model <- mlp(n_classes = ds$n_classes)
  
  model$to(device = config$device)

  optimizer <- torch::optim_adam(
    params = model$parameters,
    lr = config$lr
  )

  loss_fn <- torch::nn_cross_entropy_loss()
  
  for (epoch in seq_len(config$n_epochs)) {
    train_losses <- train(model, buffer, train_dl, optimizer, loss_fn, config)
    metrics <- valid(model, valid_dl, loss_fn, config)
    cat(sprintf("[Epoch %d] Train{Loss: %3f} Valid{Loss: %3f, Acc: %3f}\n", 
                epoch, mean(train_losses), metrics$loss, metrics$acc))
  }
  
  model
}
