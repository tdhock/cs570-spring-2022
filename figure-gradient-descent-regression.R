library(animint2)
library(data.table)
true.f.list <- list(
  constant=function(x)3,
  linear_up=function(x)2*x + 5,
  linear_down=function(x)3-x,
  quadratic=function(x)x^2,
  sin=function(x)5*sin(2*x)+5)
set.seed(1)
N <- 200
x <- runif(N, -3, 3)
max.slope <- 4
min.slope <- -3
min.intercept <- -1
max.intercept <- 7
grid.dt <- CJ(
  slope=seq(min.slope, max.slope, by=0.1),
  intercept=seq(min.intercept, max.intercept, by=0.1))
sim.dt.list <- list()
weight.dt.list <- list()
loss.dt.list <- list()
overview.dt.list <- list()
pred.dt.list <- list()
for(pattern in names(true.f.list)){
  true.f <- true.f.list[[pattern]]
  set.seed(1)
  y <- true.f(x) + rnorm(N, 0, 2)
  expr <- sub("[1] ", "", capture.output(print(body(true.f))), fixed=TRUE)
  pattern_expr <- sprintf("%s y = %s", pattern, expr)
  loss.fun <- function(intercept, slope){
    mean((0.5*(x*slope + intercept - y))^2)
  }
  grid.dt[, loss := loss.fun(intercept, slope), by=.(slope, intercept)]
  ##overview.dt.list[[pattern]] <- data.table()
  ## grad desc
  feature.mat <- cbind(1, x)
  for(step.size in 10^seq(-3, 0)){
    sim.dt.list[[paste(pattern, step.size)]] <- data.table(
      pattern, step.size, pattern_expr, x, y)
    loss.dt.list[[paste(pattern, step.size)]] <- data.table(
      pattern, step.size, grid.dt)
    weight.vec <- c(intercept=0,slope=0)
    for(iteration in 0:40){
      pred.vec <- as.numeric(feature.mat %*% weight.vec)
      resid.vec <- pred.vec-y
      pred.dt.list[[paste(pattern, step.size, iteration)]] <- data.table(
        pattern, step.size, iteration,
        x, y, prediction=pred.vec, residual=resid.vec)
      grad.vec <- colMeans(resid.vec * feature.mat)
      dir.vec <- -grad.vec * step.size
      names(dir.vec) <- names(weight.vec)
      after.weight <- weight.vec + dir.vec
      weight.dt.list[[paste(pattern, step.size, iteration)]] <- data.table(
        pattern, step.size, iteration, t(weight.vec),
        orig=t(weight.vec),
        dir=t(dir.vec),
        after=t(after.weight),
        loss=do.call(loss.fun, as.list(weight.vec)))
      weight.vec <- after.weight
    }
  }
}
sim.dt <- do.call(rbind, sim.dt.list)
ggplot()+
  facet_grid(. ~ pattern_expr)+
  geom_point(aes(
    x, y),
    data=sim.dt)

loss.dt <- do.call(rbind, loss.dt.list)
loss.dt[, relative.loss := (loss-min(loss))/(max(loss)-min(loss)), by=pattern]
loss.dt[, .(min.loss=min(loss)), by=pattern]
weight.dt <- do.call(rbind, weight.dt.list)
for(axis.name in names(weight.vec)){
  for(dir.or.not in c("", "after.")){
    var.name <- paste0(dir.or.not, axis.name)
    value.vec <- weight.dt[[var.name]]
    rep.list <- list(
      list(comp=`<`, extreme="min", replace=-Inf),
      list(comp=`>`, extreme="max", replace=Inf))
    for(info in rep.list){
      min.or.max <- get(paste0(info$extreme, ".", axis.name))
      set(
        weight.dt,
        i=which(info$comp(value.vec, min.or.max)),
        j=var.name,
        value=info$replace)
    }
  }
}
ggplot()+
  facet_grid(step.size ~ pattern)+
  geom_tile(aes(
    slope, intercept, fill=relative.loss),
    data=loss.dt)+
  scale_fill_gradient(low="white", high="red")+
  theme_bw()+
  geom_point(aes(
    slope, intercept),
    data=weight.dt)+
  geom_segment(aes(
    slope, intercept,
    xend=after.slope, yend=after.intercept),
    data=weight.dt)

some.loss <- loss.dt[, .(
  max.loss=max(loss)
), by=pattern
][
  weight.dt, on="pattern"
][
  loss<max.loss
]
some.loss <- weight.dt[, .(
  max.loss=loss[1]
), by=pattern
][
  weight.dt, on="pattern"
][
  loss > max.loss, loss := Inf
]
ggplot()+
  facet_grid(pattern ~ step.size, scales="free_y")+
  geom_line(aes(
    iteration, loss),
    size=2,
    data=some.loss)+
  theme_bw()+
  theme(panel.margin=grid::unit(0, "lines"))

dput(RColorBrewer::brewer.pal(Inf, "Blues"))
step.colors <- c(
  "1"="#08306B",
  "0.1"="#2171B5",
  "0.01"="#6BAED6",
  "0.001"="#C6DBEF")
some.loss[, Step.size := factor(step.size)]
ggplot()+
  facet_grid(pattern ~ ., scales="free_y")+
  geom_line(aes(
    iteration, loss, group=step.size, color=Step.size),
    size=2,
    data=some.loss)+
  scale_color_manual(values=step.colors, breaks=names(step.colors))+
  theme_bw()+
  theme(panel.margin=grid::unit(0, "lines"))

weight.dt[, Step.size := factor(step.size)]
ggplot()+
  scale_color_manual(values=step.colors, breaks=names(step.colors))+
  facet_grid(. ~ pattern)+
  geom_tile(aes(
    slope, intercept, fill=relative.loss),
    data=loss.dt)+
  scale_fill_gradient(low="white", high="red")+
  theme_bw()+
  geom_point(aes(
    slope, intercept, color=Step.size),
    data=weight.dt)+
  geom_segment(aes(
    slope, intercept,
    color=Step.size,
    xend=after.slope, yend=after.intercept),
    data=weight.dt)

pred.dt <- do.call(rbind, pred.dt.list)
some.loss[, pattern.step := paste(pattern, step.size)]
loss.dt[, pattern.step := paste(pattern, step.size)]
weight.dt[, pattern.step := paste(pattern, step.size)]
sim.dt[, pattern.step := paste(pattern, step.size)]
pred.dt[, pattern.step := paste(pattern, step.size)]
pred.dt[, inf.pred := ifelse(
  prediction>max(y), Inf, ifelse(
    prediction<min(y), -Inf, prediction))]
viz <- animint(
  title="Gradient descent for regression",
  data=ggplot()+
    ggtitle("Data and regression model")+
    theme_bw()+
    theme_animint(height=250, width=750)+
    geom_point(aes(
      x, y, key=x),
      data=sim.dt,
      showSelected="pattern.step")+
    scale_x_continuous(
      "input/feature")+
    scale_y_continuous(
      "output/label")+
    geom_segment(aes(
      x, y,
      key=x,
      color=line,
      xend=x, yend=inf.pred),
      data=data.table(pred.dt, line="error"),
      chunk_vars="pattern.step",
      size=1,
      showSelected=c("pattern.step", "iteration"))+
    geom_line(aes(
      x, inf.pred, key=1, color=line),
      chunk_vars="pattern.step",
      data=data.table(pred.dt, line="prediction"),
      showSelected=c("pattern.step","iteration")),
    ## geom_abline(aes(
    ##   slope=orig.slope, intercept=orig.intercept,
    ##   color=line,
    ##   key=1),
    ##   data=data.table(weight.dt, line="prediction"),
    ##   showSelected=c("pattern.step", "iteration")),
  lossIterations=ggplot()+
    ggtitle("Loss, select iteration/data/step size")+
    theme_bw()+
    theme_animint(width=350, height=350)+
    facet_grid(pattern ~ ., scales="free_y")+
    make_tallrect(some.loss, "iteration")+
    theme(axis.title=element_text(size=20))+
    geom_line(aes(
      iteration, loss, group=step.size, color=Step.size),
      size=4,
      clickSelects="pattern.step",
      data=some.loss)+
    geom_point(aes(
      iteration, loss, key=1),
      color="black",
      fill="white",
      size=5,
      showSelected=c("pattern.step", "iteration"),
      data=some.loss)+
    scale_color_manual(values=step.colors, breaks=names(step.colors))+
    theme(panel.margin=grid::unit(0, "lines")),
  lossGrid=ggplot()+
    ggtitle("Optimization variables, select iteration")+
    theme_bw()+
    theme_animint(width=350, height=350)+
    geom_tile(aes(
      slope, intercept, fill=relative.loss, key=paste(slope, intercept)),
      showSelected="pattern.step",
      data=loss.dt)+
    scale_fill_gradient(low="white", high="red")+
    geom_point(aes(
      slope, intercept, key=iteration),
      showSelected="pattern.step",
      clickSelects="iteration",
      size=5,
      alpha=0.55,
      data=weight.dt)+
    geom_point(aes(
      slope, intercept, key=1),
      showSelected=c("pattern.step", "iteration"),
      fill="white",
      color="black",
      size=5,
      data=weight.dt)+
    geom_segment(aes(
      slope, intercept,
      key=1,
      xend=after.slope, yend=after.intercept),
      showSelected=c("pattern.step", "iteration"),
      color="deepskyblue",
      data=weight.dt)+
    geom_point(aes(
      after.slope, after.intercept,
      key=1),
      showSelected=c("pattern.step", "iteration"),
      color="deepskyblue",
      data=weight.dt),
  first=list(
    pattern.step="linear_up 0.1"),
  time=list(
    variable="iteration",
    ms=500),
  duration=list(
    pattern.step=500,
    iteration=500),
  out.dir="figure-gradient-descent-regression")

