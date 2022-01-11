if(!file.exists("ESL.mixture.rda")){
  download.file(
    "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/ESL.mixture.rda",
    "ESL.mixture.rda")
}
load("ESL.mixture.rda")
str(ESL.mixture)
library(data.table)
mixture.dt <- with(ESL.mixture, data.table(y, x))
fwrite(mixture.dt, "ESL.mixture.csv", col.names=FALSE)
