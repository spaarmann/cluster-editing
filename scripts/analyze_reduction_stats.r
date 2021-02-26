RUN = 'data/runs/full/59da0f6_2021-02-25T17-05-08+01-00/'

full = data.frame(row.names = c("k", "red"))
fast = data.frame(row.names = c("k", "red"))
param_dep = data.frame(row.names = c("k", "red"))

for (stat_folder in list.files(paste(RUN, "runner_out/", sep=""), full.names = TRUE)) {
  if (!dir.exists(stat_folder)) {
    next
  }
  
  for (file in list.files(stat_folder, full.names = TRUE)) {
    if (endsWith(file, "_full.stats")) {
      file_data = read.table(file, header = TRUE, sep = "|")
      full = rbind(full, file_data)
    }
    else if (endsWith(file, "_fast.stats")) {
      file_data = read.table(file, header = TRUE, sep = "|")
      fast = rbind(fast, file_data)
    }
    else if (endsWith(file, "_paramdep.stats")) {
      file_data = read.table(file, header = TRUE, sep = "|")
      param_dep = rbind(param_dep, file_data)
    }
  }
}

# x-axis: remaining cost
#  - alternative idea: spent cost?
# y-axis: reduction effectiveness

# just throw in every data point for now I think