# ============================================================================
# ENV_nat.R - Environmental Data Analysis: City to National Level
# Project: Urban Productive Safety Net and Jobs Project (UPSNJP)
# Author: Belachew Hirpa Lemma
# Purpose: Aggregate and analyze environmental compliance data from city-level
#          sources to produce national-level insights and reports
# ============================================================================

# ============================================================================
# 1. LOAD REQUIRED LIBRARIES
# ============================================================================

library(tidyverse)      # Data manipulation and visualization
library(readxl)         # Read Excel files
library(lubridate)      # Work with dates
library(scales)         # Format chart axes
library(knitr)          # Generate tables for reporting
library(ggplot2)        # Advanced visualizations

# ============================================================================
# 2. SETUP AND CONFIGURATION
# ============================================================================

# Set working directory (update this to your file path)
# setwd("C:/Users/Belachew/UPSNJP-Data")

# Define file paths for city-level data
city_data_files <- list(
  addis_ababa = "data/addis_ababa_env.csv",
  dire_dawa = "data/dire_dawa_env.csv",
  bahir_dar = "data/bahir_dar_env.csv",
  hawassa = "data/hawassa_env.csv",
  jimma = "data/jimma_env.csv"
)

# Define compliance thresholds (World Bank standards)
compliance_thresholds <- list(
  environmental_impact_score = 75,    # Minimum acceptable score (%)
  safeguard_compliance_rate = 80,      # Minimum acceptable rate (%)
  grievance_redress_rate = 90,         # Minimum acceptable rate (%)
  community_consultation_score = 70    # Minimum acceptable score (%)
)

# ============================================================================
# 3. FUNCTION TO LOAD AND CLEAN CITY-LEVEL DATA
# ============================================================================

load_city_data <- function(file_path, city_name) {
  # Check if file exists
  if (!file.exists(file_path)) {
    warning(paste("File not found:", file_path, "- Creating sample data for", city_name))
    return(create_sample_data(city_name))
  }
  
  # Load data
  data <- read_csv(file_path, show_col_types = FALSE)
  
  # Add city column and standardize column names
  data <- data %>%
    mutate(
      city = city_name,
      date = as.Date(date, format = "%Y-%m-%d"),
      environmental_score = as.numeric(environmental_score),
      safeguard_compliance = as.numeric(safeguard_compliance),
      ohs_compliance = as.numeric(ohs_compliance),
      grievance_redressed = as.numeric(grievance_redressed),
      grievance_total = as.numeric(grievance_total),
      consultations_held = as.numeric(consultations_held),
      consultations_planned = as.numeric(consultations_planned)
    ) %>%
    # Calculate derived metrics
    mutate(
      grievance_redress_rate = ifelse(grievance_total > 0, 
                                      (grievance_redressed / grievance_total) * 100, 
                                      NA),
      consultation_completion_rate = ifelse(consultations_planned > 0,
                                            (consultations_held / consultations_planned) * 100,
                                            NA),
      overall_compliance = (environmental_score + safeguard_compliance + ohs_compliance) / 3
    )
  
  return(data)
}

# ============================================================================
# 4. FUNCTION TO CREATE SAMPLE DATA (for demonstration)
# ============================================================================

create_sample_data <- function(city_name) {
  set.seed(123)  # For reproducibility
  
  dates <- seq(as.Date("2024-01-01"), as.Date("2024-12-31"), by = "month")
  
  data <- data.frame(
    date = dates,
    environmental_score = round(runif(length(dates), 65, 95), 1),
    safeguard_compliance = round(runif(length(dates), 70, 98), 1),
    ohs_compliance = round(runif(length(dates), 60, 95), 1),
    grievance_redressed = sample(5:50, length(dates), replace = TRUE),
    grievance_total = grievance_redressed + sample(-10:20, length(dates), replace = TRUE),
    consultations_held = sample(10:60, length(dates), replace = TRUE),
    consultations_planned = consultations_held + sample(-5:15, length(dates), replace = TRUE)
  )
  
  # Ensure grievance_total >= grievance_redressed
  data$grievance_total <- pmax(data$grievance_total, data$grievance_redressed)
  data$consultations_planned <- pmax(data$consultations_planned, data$consultations_held)
  
  data$city <- city_name
  
  return(data)
}

# ============================================================================
# 5. LOAD AND INTEGRATE DATA FROM ALL CITIES
# ============================================================================

# Load all city datasets (using sample data if files not found)
cat("Loading environmental data from all cities...\n")

all_cities_data <- bind_rows(
  load_city_data(city_data_files$addis_ababa, "Addis Ababa"),
  load_city_data(city_data_files$dire_dawa, "Dire Dawa"),
  load_city_data(city_data_files$bahir_dar, "Bahir Dar"),
  load_city_data(city_data_files$hawassa, "Hawassa"),
  load_city_data(city_data_files$jimma, "Jimma")
)

cat("Data loaded successfully!\n")
cat("Total records:", nrow(all_cities_data), "\n")
cat("Cities included:", paste(unique(all_cities_data$city), collapse = ", "), "\n\n")

# ============================================================================
# 6. NATIONAL-LEVEL AGGREGATION
# ============================================================================

# Aggregate to national level
national_summary <- all_cities_data %>%
  summarise(
    # Environmental metrics
    avg_environmental_score = mean(environmental_score, na.rm = TRUE),
    avg_safeguard_compliance = mean(safeguard_compliance, na.rm = TRUE),
    avg_ohs_compliance = mean(ohs_compliance, na.rm = TRUE),
    avg_overall_compliance = mean(overall_compliance, na.rm = TRUE),
    
    # Grievance metrics
    total_grievances = sum(grievance_total, na.rm = TRUE),
    total_grievances_redressed = sum(grievance_redressed, na.rm = TRUE),
    national_grievance_redress_rate = (total_grievances_redressed / total_grievances) * 100,
    
    # Consultation metrics
    total_consultations_held = sum(consultations_held, na.rm = TRUE),
    total_consultations_planned = sum(consultations_planned, na.rm = TRUE),
    national_consultation_rate = (total_consultations_held / total_consultations_planned) * 100
  )

# ============================================================================
# 7. CITY-LEVEL COMPARISON ANALYSIS
# ============================================================================

city_summary <- all_cities_data %>%
  group_by(city) %>%
  summarise(
    avg_environmental_score = mean(environmental_score, na.rm = TRUE),
    avg_safeguard_compliance = mean(safeguard_compliance, na.rm = TRUE),
    avg_ohs_compliance = mean(ohs_compliance, na.rm = TRUE),
    avg_overall_compliance = mean(overall_compliance, na.rm = TRUE),
    grievance_redress_rate = mean(grievance_redress_rate, na.rm = TRUE),
    consultation_completion_rate = mean(consultation_completion_rate, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(avg_overall_compliance))

# Identify cities below compliance thresholds
below_threshold <- city_summary %>%
  filter(avg_overall_compliance < compliance_thresholds$environmental_impact_score) %>%
  select(city, avg_overall_compliance)

# ============================================================================
# 8. TIME-SERIES TREND ANALYSIS
# ============================================================================

monthly_trends <- all_cities_data %>%
  group_by(city, month = floor_date(date, "month")) %>%
  summarise(
    avg_environmental_score = mean(environmental_score, na.rm = TRUE),
    .groups = "drop"
  )

# National monthly trend
national_monthly <- all_cities_data %>%
  group_by(month = floor_date(date, "month")) %>%
  summarise(
    national_avg_score = mean(environmental_score, na.rm = TRUE),
    .groups = "drop"
  )

# ============================================================================
# 9. GENERATE VISUALIZATIONS
# ============================================================================

# Visualization 1: City comparison bar chart
plot_city_comparison <- ggplot(city_summary, aes(x = reorder(city, avg_overall_compliance), 
                                                   y = avg_overall_compliance, 
                                                   fill = avg_overall_compliance >= 75)) +
  geom_bar(stat = "identity") +
  geom_hline(yintercept = 75, linetype = "dashed", color = "red", linewidth = 1) +
  scale_fill_manual(values = c("TRUE" = "#2E8B57", "FALSE" = "#CD5C5C"), 
                    labels = c("FALSE" = "Below Threshold", "TRUE" = "Compliant")) +
  coord_flip() +
  labs(
    title = "Environmental Compliance by City",
    subtitle = "UPSNJP Monitoring Data (World Bank Standards)",
    x = "City",
    y = "Overall Compliance Score (%)",
    fill = "Status"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 10, color = "gray50"),
    legend.position = "bottom"
  )

# Visualization 2: Monthly trends line chart
plot_monthly_trends <- ggplot(monthly_trends, aes(x = month, y = avg_environmental_score, color = city)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  geom_hline(yintercept = 75, linetype = "dashed", color = "red", alpha = 0.7) +
  labs(
    title = "Monthly Environmental Score Trends by City",
    subtitle = "January - December 2024",
    x = "Month",
    y = "Environmental Score (%)",
    color = "City"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

# Visualization 3: Grievance redress rate by city
plot_grievance <- ggplot(city_summary, aes(x = reorder(city, grievance_redress_rate), 
                                            y = grievance_redress_rate,
                                            fill = grievance_redress_rate >= 90)) +
  geom_bar(stat = "identity") +
  geom_hline(yintercept = 90, linetype = "dashed", color = "red", linewidth = 1) +
  coord_flip() +
  labs(
    title = "Grievance Redress Rate by City",
    subtitle = "Target: 90%",
    x = "City",
    y = "Redress Rate (%)"
  ) +
  theme_minimal()

# ============================================================================
# 10. GENERATE REPORT TABLES
# ============================================================================

# National summary table
national_table <- national_summary %>%
  select(
    `Environmental Score (%)` = avg_environmental_score,
    `Safeguard Compliance (%)` = avg_safeguard_compliance,
    `OHS Compliance (%)` = avg_ohs_compliance,
    `Overall Compliance (%)` = avg_overall_compliance,
    `Grievance Redress Rate (%)` = national_grievance_redress_rate,
    `Consultation Rate (%)` = national_consultation_rate
  )

# City ranking table
city_ranking_table <- city_summary %>%
  select(
    City = city,
    `Env Score` = avg_environmental_score,
    `Safeguard` = avg_safeguard_compliance,
    `OHS` = avg_ohs_compliance,
    `Overall` = avg_overall_compliance,
    `Grievance Rate` = grievance_redress_rate,
    `Consultation Rate` = consultation_completion_rate
  )

# ============================================================================
# 11. IDENTIFY GAPS AND DISPARITIES
# ============================================================================

cat("\n" , "=", 60, "\n", sep="")
cat("ENVIRONMENTAL COMPLIANCE REPORT - NATIONAL LEVEL\n")
cat("=", 60, "\n\n", sep="")

# Calculate disparities
score_range <- diff(range(city_summary$avg_overall_compliance))
lowest_city <- city_summary %>% filter(avg_overall_compliance == min(avg_overall_compliance))
highest_city <- city_summary %>% filter(avg_overall_compliance == max(avg_overall_compliance))

cat("Key Findings:\n")
cat("--------------\n")
cat("1. National average compliance:", round(national_summary$avg_overall_compliance, 1), "%\n")
cat("2. Range between cities:", round(score_range, 1), "%\n")
cat("3. Highest performing city:", highest_city$city, "(", round(highest_city$avg_overall_compliance, 1), "%)\n")
cat("4. Lowest performing city:", lowest_city$city, "(", round(lowest_city$avg_overall_compliance, 1), "%)\n")

if(nrow(below_threshold) > 0) {
  cat("\nCities BELOW compliance threshold (75%):\n")
  for(i in 1:nrow(below_threshold)) {
    cat("   -", below_threshold$city[i], ":", round(below_threshold$avg_overall_compliance[i], 1), "%\n")
  }
} else {
  cat("\nAll cities meet or exceed the compliance threshold.\n")
}

# ============================================================================
# 12. EXPORT RESULTS
# ============================================================================

# Create output directory if it doesn't exist
if(!dir.exists("output")) {
  dir.create("output")
}

# Save summary tables as CSV
write_csv(city_summary, "output/city_level_summary.csv")
write_csv(national_summary, "output/national_summary.csv")
write_csv(all_cities_data, "output/consolidated_city_data.csv")

# Save visualizations
ggsave("output/city_comparison.png", plot_city_comparison, width = 10, height = 6, dpi = 300)
ggsave("output/monthly_trends.png", plot_monthly_trends, width = 12, height = 6, dpi = 300)
ggsave("output/grievance_analysis.png", plot_grievance, width = 10, height = 6, dpi = 300)

cat("\n")
cat("=", 60, "\n", sep="")
cat("OUTPUT FILES GENERATED:\n")
cat("=", 60, "\n", sep="")
cat("1. output/city_level_summary.csv - City comparison data\n")
cat("2. output/national_summary.csv - National aggregates\n")
cat("3. output/consolidated_city_data.csv - All raw data combined\n")
cat("4. output/city_comparison.png - Bar chart visualization\n")
cat("5. output/monthly_trends.png - Time-series chart\n")
cat("6. output/grievance_analysis.png - Grievance analysis chart\n")

# ============================================================================
# 13. PRINT FINAL TABLES TO CONSOLE
# ============================================================================

cat("\n")
cat("=", 60, "\n", sep="")
cat("NATIONAL SUMMARY TABLE\n")
cat("=", 60, "\n", sep="")
print(national_table)

cat("\n")
cat("=", 60, "\n", sep="")
cat("CITY RANKING TABLE (by Overall Compliance)\n")
cat("=", 60, "\n", sep="")
print(city_ranking_table)

# ============================================================================
# END OF SCRIPT
# ============================================================================
