create_km_plot <- function(surv_object, data, title, filename, legend_labs = NULL,
                           palette = c("#0073C2FF", "#EFC000FF", "#FF0000"),
                           output_dir = "figures", width = 10, height = 10) {
  
  # Ensure the output directory exists
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # Determine number of groups in survival object
  num_groups <- if (!is.null(surv_object$strata)) length(surv_object$strata) else 1
  
  # Auto-coordinate for p-value in single-group scenarios
  pval_coord <- if(num_groups == 1) {
    c(max(data$LoS, na.rm = TRUE) * 0.8, 0.8)
  } else {
    NULL
  }
  
  # Enable p-value display only if comparing groups
  pval_flag <- (num_groups > 1)
  
  # Use ggsurvplot's built-in capabilities for proper alignment
  # This is more reliable than manually arranging plots
  km_plot <- ggsurvplot(
    surv_object, data = data,
    palette = palette,
    pval = pval_flag,
    conf.int = TRUE,
    pval.method = pval_flag,
    surv.median.line = "hv",
    risk.table = TRUE,
    censor.size = 0.8,
    font.main = 10,
    log.rank.weights = "1",
    pval.method.size = 3,
    pval.size = 3,
    legend.title = "",
    fontsize = 3,
    conf.int.style = "ribbon",
    pval.coord = pval_coord,
    legend.labs = legend_labs,
    xlab = "Survival time in days",
    ylab = "Survival Probability",
    title = title,
    tables.theme = theme_cleantable(),
    risk.table.height = 0.15,  # Control risk table height (25% of total)
    risk.table.title = "Number at Risk",
    tables.font = 2,
    risk.table.fontsize = 3,
    risk.table.y.text.col = FALSE,
    # Critical parameter for alignment:
    tables.y.text = TRUE,  # Remove strata names from risk table to save space
    ggtheme = theme_bw()    # Clean theme for better visibility
  )
  
  # Save the plot without risk table
  plot_filepath <- file.path(output_dir, filename)
  #ggsave(plot_filepath, km_plot$plot, width = width, height = height)
  
  # Save the combined plot using survminer's built-in arrangement
  # This ensures proper alignment of axes
  combined_filename <- gsub("\\.png$", "_with_table.png", filename)
  combined_filepath <- file.path(output_dir, combined_filename)
  
  # Use ggsurvplot's arrange method which handles alignment properly
  km_plot_arranged <- arrange_ggsurvplots(
    list(km_plot),
    print = FALSE,
    ncol = 1, nrow = 1
  )
  
  # Save the arranged plot
  ggsave(combined_filepath, km_plot_arranged, width = width, height = height)
  
  return(km_plot)
}
