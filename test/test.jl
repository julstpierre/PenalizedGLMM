const datadir = "data"
const covfile = datadir * "/covariate.txt"
const plinkfile = datadir * "/UKBB"
const grmfile = datadir * "/grm.txt.gz"

pglmm(@formula(y ~ SEX + AGE), covfile, plinkfile, grmfile)