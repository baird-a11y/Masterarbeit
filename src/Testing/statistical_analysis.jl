# =============================================================================
# ROBUSTE STATISTISCHE ANALYSE FÜR KLEINE STICHPROBEN
# =============================================================================
# Speichern als: robust_statistical_analysis.jl

using Statistics
using StatsBase
using LinearAlgebra
using Printf

"""
Struktur für statistische Analyse-Ergebnisse
"""
struct RobustStatisticalAnalysis
    descriptive_stats::Dict
    confidence_intervals::Dict
    trend_analysis::Dict
    significance_tests::Dict
    sample_sizes::Dict
end

"""
Führt robuste statistische Analyse für kleine Stichproben durch
"""
function perform_robust_statistical_analysis(batch_results; confidence_level=0.95, verbose=true)
    
    if verbose
        println("=== ROBUSTE STATISTISCHE SKALIERUNGS-ANALYSE ===")
        println("Konfidenz-Niveau: $(confidence_level*100)%")
    end
    
    # Sammle Daten pro Kristallanzahl
    data_by_crystal = Dict{Int, Vector{Float64}}()
    
    for (n_crystals, results) in batch_results.results_per_crystal
        mae_values = Float64[]
        for result in results
            if isfinite(result.mae_total) && result.mae_total > 0
                push!(mae_values, result.mae_total)
            end
        end
        if !isempty(mae_values)
            data_by_crystal[n_crystals] = mae_values
        end
    end
    
    # 1. Deskriptive Statistiken
    if verbose
        println("Berechne deskriptive Statistiken...")
    end
    descriptive = calculate_robust_descriptive_stats(data_by_crystal)
    
    # 2. Konfidenzintervalle (Bootstrap für kleine Stichproben)
    if verbose
        println("Berechne Bootstrap-Konfidenzintervalle ($(confidence_level*100)%)...")
    end
    confidence = calculate_bootstrap_confidence_intervals(data_by_crystal, confidence_level)
    
    # 3. Trend-Analyse
    if verbose
        println("Führe Trend-Analyse durch...")
    end
    trends = calculate_robust_trends(data_by_crystal)
    
    # 4. Signifikanz-Tests (angepasst für kleine Stichproben)
    if verbose
        println("Führe robuste Signifikanz-Tests durch...")
    end
    significance = perform_robust_significance_tests(data_by_crystal)
    
    # 5. Sample-Größen
    sample_sizes = Dict(k => length(v) for (k, v) in data_by_crystal)
    
    analysis = RobustStatisticalAnalysis(
        descriptive,
        confidence,
        trends,
        significance,
        sample_sizes
    )
    
    if verbose
        print_robust_statistical_summary(analysis)
    end
    
    return analysis
end

"""
Berechnet robuste deskriptive Statistiken
"""
function calculate_robust_descriptive_stats(data_by_crystal)
    stats = Dict{Int, Dict{String, Float64}}()
    
    for (n_crystals, values) in data_by_crystal
        if isempty(values)
            continue
        end
        
        # Robuste Statistiken verwenden
        stats[n_crystals] = Dict(
            "mean" => mean(values),
            "median" => median(values),
            "std" => length(values) > 1 ? std(values) : 0.0,
            "mad" => mad(values),  # Median Absolute Deviation
            "min" => minimum(values),
            "max" => maximum(values),
            "q25" => quantile(values, 0.25),
            "q75" => quantile(values, 0.75),
            "iqr" => quantile(values, 0.75) - quantile(values, 0.25),
            "n" => Float64(length(values))
        )
    end
    
    return stats
end

"""
Berechnet Bootstrap-Konfidenzintervalle für kleine Stichproben
"""
function calculate_bootstrap_confidence_intervals(data_by_crystal, confidence_level; n_bootstrap=1000)
    intervals = Dict{Int, Dict{String, Tuple{Float64, Float64}}}()
    
    for (n_crystals, values) in data_by_crystal
        if isempty(values)
            continue
        end
        
        # Bootstrap für Mean
        bootstrap_means = Float64[]
        n_samples = length(values)
        
        if n_samples == 1
            # Einzelner Wert: Kein Konfidenzintervall möglich
            intervals[n_crystals] = Dict(
                "mean" => (values[1], values[1]),
                "median" => (values[1], values[1])
            )
        else
            # Bootstrap-Sampling
            for _ in 1:n_bootstrap
                bootstrap_sample = sample(values, n_samples, replace=true)
                push!(bootstrap_means, mean(bootstrap_sample))
            end
            
            # Perzentil-Methode für Konfidenzintervall
            alpha = (1.0 - confidence_level) / 2.0
            lower_percentile = quantile(bootstrap_means, alpha)
            upper_percentile = quantile(bootstrap_means, 1.0 - alpha)
            
            intervals[n_crystals] = Dict(
                "mean" => (lower_percentile, upper_percentile),
                "median" => (quantile(values, alpha), quantile(values, 1.0 - alpha))
            )
        end
    end
    
    return intervals
end

"""
Berechnet robuste Trends ohne Annahmen über Verteilung
"""
function calculate_robust_trends(data_by_crystal)
    crystal_counts = sort(collect(keys(data_by_crystal)))
    
    if length(crystal_counts) < 2
        return Dict("trend_type" => "insufficient_data")
    end
    
    # Verwende Mediane für robuste Trend-Schätzung
    medians = [median(data_by_crystal[k]) for k in crystal_counts]
    means = [mean(data_by_crystal[k]) for k in crystal_counts]
    
    # Spearman-Rangkorrelation (robuster als Pearson)
    n = length(crystal_counts)
    ranks_x = collect(1:n)
    ranks_y = sortperm(sortperm(medians))
    
    spearman_corr = cor(Float64.(ranks_x), Float64.(ranks_y))
    
    # Einfache lineare Regression mit Medianen
    X = hcat(ones(n), Float64.(crystal_counts))
    y = medians
    
    # Least Squares für Trend-Linie
    if n >= 2
        coeffs = X \ y
        intercept, slope = coeffs[1], coeffs[2]
        
        # R² berechnen
        y_pred = X * coeffs
        ss_res = sum((y .- y_pred).^2)
        ss_tot = sum((y .- mean(y)).^2)
        r_squared = ss_tot > 0 ? 1 - ss_res/ss_tot : 0.0
    else
        intercept, slope, r_squared = 0.0, 0.0, 0.0
    end
    
    # Trend-Interpretation
    if abs(slope) < 0.001
        trend_type = "constant"
    elseif slope > 0
        trend_type = "increasing"
    else
        trend_type = "decreasing"
    end
    
    return Dict(
        "trend_type" => trend_type,
        "slope" => slope,
        "intercept" => intercept,
        "r_squared" => r_squared,
        "spearman_correlation" => spearman_corr,
        "median_progression" => medians,
        "mean_progression" => means
    )
end

"""
Führt robuste Signifikanz-Tests für kleine Stichproben durch
"""
function perform_robust_significance_tests(data_by_crystal)
    results = Dict{String, Any}()
    crystal_counts = sort(collect(keys(data_by_crystal)))
    
    # Kruskal-Wallis Test (nicht-parametrisch, robust für kleine Stichproben)
    if length(crystal_counts) >= 2
        all_values = Float64[]
        groups = Int[]
        
        for (i, k) in enumerate(crystal_counts)
            values = data_by_crystal[k]
            append!(all_values, values)
            append!(groups, fill(i, length(values)))
        end
        
        if length(unique(groups)) >= 2 && length(all_values) >= 3
            # Vereinfachter Kruskal-Wallis H-Statistik
            h_stat = calculate_kruskal_wallis_h(all_values, groups)
            results["kruskal_wallis_h"] = h_stat
            results["kruskal_wallis_significant"] = h_stat > 5.991  # Chi-Quadrat kritischer Wert für α=0.05, df=2
        else
            results["kruskal_wallis_h"] = NaN
            results["kruskal_wallis_significant"] = false
        end
    end
    
    # Paarweise Vergleiche mit Mann-Whitney U Test (robust für kleine Stichproben)
    pairwise = Dict{String, Dict}()
    
    for i in 1:length(crystal_counts)-1
        k1, k2 = crystal_counts[i], crystal_counts[i+1]
        
        if haskey(data_by_crystal, k1) && haskey(data_by_crystal, k2)
            values1 = data_by_crystal[k1]
            values2 = data_by_crystal[k2]
            
            if !isempty(values1) && !isempty(values2)
                # Mann-Whitney U Test
                u_stat, p_value = mann_whitney_u_test(values1, values2)
                
                pairwise["$(k1)_vs_$(k2)"] = Dict(
                    "u_statistic" => u_stat,
                    "p_value" => p_value,
                    "significant" => p_value < 0.05,
                    "effect_size" => calculate_effect_size(values1, values2)
                )
            end
        end
    end
    
    results["pairwise_comparisons"] = pairwise
    
    return results
end

"""
Vereinfachter Kruskal-Wallis H-Test
"""
function calculate_kruskal_wallis_h(values, groups)
    n = length(values)
    if n < 3
        return NaN
    end
    
    # Ränge berechnen
    ranks = sortperm(sortperm(values))
    
    # Summe der Ränge pro Gruppe
    unique_groups = unique(groups)
    rank_sums = Dict{Int, Float64}()
    group_sizes = Dict{Int, Int}()
    
    for g in unique_groups
        group_indices = findall(x -> x == g, groups)
        rank_sums[g] = sum(ranks[group_indices])
        group_sizes[g] = length(group_indices)
    end
    
    # H-Statistik berechnen
    h = 12.0 / (n * (n + 1))
    sum_term = 0.0
    
    for g in unique_groups
        if group_sizes[g] > 0
            sum_term += (rank_sums[g]^2) / group_sizes[g]
        end
    end
    
    h = h * sum_term - 3 * (n + 1)
    
    return h
end

"""
Vereinfachter Mann-Whitney U Test
"""
function mann_whitney_u_test(values1, values2)
    n1, n2 = length(values1), length(values2)
    
    if n1 == 0 || n2 == 0
        return NaN, 1.0
    end
    
    # Kombiniere und ranke
    combined = vcat(values1, values2)
    ranks = sortperm(sortperm(combined))
    
    # Summe der Ränge für Gruppe 1
    r1 = sum(ranks[1:n1])
    
    # U-Statistik
    u1 = r1 - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    u = min(u1, u2)
    
    # Approximative p-Wert Berechnung (für kleine Stichproben ungenau)
    if n1 >= 3 && n2 >= 3
        # Normal-Approximation
        mu_u = n1 * n2 / 2
        sigma_u = sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        
        if sigma_u > 0
            z = (u - mu_u) / sigma_u
            # Vereinfachter zweiseitiger p-Wert
            p_value = 2 * (1 - normcdf(abs(z)))
        else
            p_value = 1.0
        end
    else
        # Für sehr kleine Stichproben: konservativ
        p_value = u < min(n1, n2) ? 0.1 : 0.5
    end
    
    return u, p_value
end

"""
Berechnet Effektgröße (Cohen's d oder Cliff's Delta für kleine Stichproben)
"""
function calculate_effect_size(values1, values2)
    n1, n2 = length(values1), length(values2)
    
    if n1 == 0 || n2 == 0
        return NaN
    end
    
    if n1 == 1 || n2 == 1
        # Cliff's Delta für sehr kleine Stichproben
        greater = 0
        less = 0
        
        for v1 in values1
            for v2 in values2
                if v1 > v2
                    greater += 1
                elseif v1 < v2
                    less += 1
                end
            end
        end
        
        delta = (greater - less) / (n1 * n2)
        return delta
    else
        # Cohen's d für größere Stichproben
        mean1, mean2 = mean(values1), mean(values2)
        pooled_std = sqrt(((n1-1)*var(values1) + (n2-1)*var(values2)) / (n1 + n2 - 2))
        
        if pooled_std > 0
            return (mean1 - mean2) / pooled_std
        else
            return 0.0
        end
    end
end

"""
Vereinfachte Normal-CDF Approximation
"""
function normcdf(z)
    # Vereinfachte Approximation der Standard-Normal-CDF
    if z < -6
        return 0.0
    elseif z > 6
        return 1.0
    else
        # Abramowitz und Stegun Approximation
        t = 1.0 / (1.0 + 0.2316419 * abs(z))
        d = 0.3989423 * exp(-z * z / 2)
        c = d * t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
        
        return z >= 0 ? 1.0 - c : c
    end
end

"""
Gibt robuste statistische Zusammenfassung aus
"""
function print_robust_statistical_summary(analysis::RobustStatisticalAnalysis)
    println("\n" * "="^80)
    println("ROBUSTE STATISTISCHE ZUSAMMENFASSUNG")
    println("="^80)
    
    # Sample-Größen
    println("\n📊 STICHPROBENGRÖSZEN:")
    for k in sort(collect(keys(analysis.sample_sizes)))
        n = analysis.sample_sizes[k]
        println("  $k Kristalle: $n Samples")
    end
    
    # Deskriptive Statistiken
    println("\n📈 DESKRIPTIVE STATISTIKEN (MAE):")
    println(@sprintf("%-10s %-10s %-10s %-10s %-10s", "Kristalle", "Mean", "Median", "Std", "MAD"))
    println("-"^50)
    
    # KORRIGIERT: Sortiere Keys, nicht die Dictionary-Paare
    for k in sort(collect(keys(analysis.descriptive_stats)))
        stats = analysis.descriptive_stats[k]
        println(@sprintf("%-10d %-10.4f %-10.4f %-10.4f %-10.4f", 
                k, stats["mean"], stats["median"], stats["std"], stats["mad"]))
    end
    
    # Konfidenzintervalle
    println("\n🎯 BOOTSTRAP-KONFIDENZINTERVALLE (95%):")
    println(@sprintf("%-10s %-25s %-25s", "Kristalle", "Mean CI", "Median CI"))
    println("-"^60)
    
    # KORRIGIERT: Sortiere Keys
    for k in sort(collect(keys(analysis.confidence_intervals)))
        intervals = analysis.confidence_intervals[k]
        mean_ci = intervals["mean"]
        median_ci = intervals["median"]
        println(@sprintf("%-10d [%.4f, %.4f]      [%.4f, %.4f]", 
                k, mean_ci[1], mean_ci[2], median_ci[1], median_ci[2]))
    end
    
    # Trend-Analyse
    println("\n📉 TREND-ANALYSE:")
    trend = analysis.trend_analysis
    if haskey(trend, "trend_type") && trend["trend_type"] != "insufficient_data"
        println("  Trend-Typ: $(trend["trend_type"])")
        println("  Steigung: $(round(trend["slope"], digits=6))")
        println("  R²: $(round(trend["r_squared"], digits=4))")
        println("  Spearman-Korrelation: $(round(trend["spearman_correlation"], digits=4))")
    else
        println("  Unzureichende Daten für Trend-Analyse")
    end
    
    # Signifikanz-Tests
    println("\n🔬 SIGNIFIKANZ-TESTS:")
    sig = analysis.significance_tests
    
    if haskey(sig, "kruskal_wallis_h") && !isnan(sig["kruskal_wallis_h"])
        println("  Kruskal-Wallis H: $(round(sig["kruskal_wallis_h"], digits=3))")
        println("  Signifikant unterschiedlich: $(sig["kruskal_wallis_significant"] ? "Ja" : "Nein")")
    end
    
    if haskey(sig, "pairwise_comparisons") && !isempty(sig["pairwise_comparisons"])
        println("\n  Paarweise Vergleiche (Mann-Whitney U):")
        for (comparison, result) in sig["pairwise_comparisons"]
            if !isnan(result["u_statistic"])
                println("    $comparison: U=$(round(result["u_statistic"], digits=2)), " *
                       "p=$(round(result["p_value"], digits=4)), " *
                       "Effektgröße=$(round(result["effect_size"], digits=3))")
            end
        end
    end
    
    println("\n" * "="^80)
end

println("Robuste Statistische Analyse Module geladen!")
println("Verwende: perform_robust_statistical_analysis(batch_results)")
println("")
println("Dieses Modul ist optimiert für kleine Stichproben und verwendet:")
println("  - Bootstrap für Konfidenzintervalle")
println("  - Nicht-parametrische Tests (Kruskal-Wallis, Mann-Whitney U)")
println("  - Robuste Statistiken (Median, MAD)")
println("  - Spearman-Rangkorrelation für Trends")