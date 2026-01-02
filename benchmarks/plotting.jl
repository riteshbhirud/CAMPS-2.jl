# CAMPS.jl/benchmarks/plotting.jl
# Publication-quality figure generation using CairoMakie
#
# Generates figures for Quantum journal publication

# Note: CairoMakie must be installed separately
# Run: ] add CairoMakie

include("benchmark_types.jl")

using Printf
using Statistics

# Try to load CairoMakie, provide fallback if not available
const MAKIE_AVAILABLE = try
    @eval using CairoMakie
    true
catch
    @warn "CairoMakie not available. Install with: ] add CairoMakie"
    false
end

# Publication figure settings
const FIGURE_WIDTH_SINGLE = 8.9   # cm, single column
const FIGURE_WIDTH_DOUBLE = 18.0  # cm, double column
const DPI = 300
const FONT_SIZE = 10
const FONT_FAMILY = "Arial"

# Colorblind-friendly color scheme (Paul Tol's palette)
const COLORS = Dict(
    :ofd => "#0077BB",       # Blue
    :obd => "#EE7733",       # Orange
    :hybrid => "#009988",    # Teal
    :none => "#CC3311",      # Red
    :predicted => "#33BBEE", # Cyan
    :actual => "#EE3377",    # Magenta
    :independent => "#0077BB",
    :entangled => "#EE7733",
    :random => "#009988",
    :qft => "#CC3311"
)

# Marker styles
const MARKERS = Dict(
    :ofd => :circle,
    :obd => :rect,
    :hybrid => :diamond,
    :none => :utriangle,
    :predicted => :star5,
    :actual => :circle
)

"""
    setup_makie_theme()

Set up publication-quality theme for CairoMakie.
"""
function setup_makie_theme()
    if !MAKIE_AVAILABLE
        @warn "CairoMakie not available"
        return
    end

    set_theme!(
        fontsize = FONT_SIZE,
        font = FONT_FAMILY,
        Axis = (
            xlabelsize = FONT_SIZE + 1,
            ylabelsize = FONT_SIZE + 1,
            xticklabelsize = FONT_SIZE - 1,
            yticklabelsize = FONT_SIZE - 1,
            titlesize = FONT_SIZE + 2,
            spinewidth = 0.5,
            xgridwidth = 0.5,
            ygridwidth = 0.5,
            xminorgridvisible = false,
            yminorgridvisible = false
        ),
        Legend = (
            labelsize = FONT_SIZE - 1,
            framewidth = 0.5,
            padding = (5, 5, 3, 3)
        ),
        Lines = (
            linewidth = 1.5
        ),
        Scatter = (
            markersize = 8
        )
    )
end

"""
    figure_gf2_validation(results::Vector{GF2ValidationResult}; save_path=nothing)

Generate Figure 1: GF(2) prediction accuracy scatter plot.

Shows predicted vs actual bond dimension with identity line.
Points on the line indicate exact prediction.
"""
function figure_gf2_validation(results::Vector{GF2ValidationResult}; save_path=nothing)
    if !MAKIE_AVAILABLE
        println("CairoMakie not available. Generating text summary instead.")
        print_gf2_summary(results)
        return nothing
    end

    setup_makie_theme()

    # Extract data by circuit type
    circuit_types = unique(r.circuit_type for r in results)

    fig = Figure(size = (400, 400))
    ax = Axis(fig[1, 1],
        xlabel = "Predicted χ",
        ylabel = "Actual χ",
        title = "GF(2) Bond Dimension Prediction Accuracy",
        aspect = 1
    )

    # Find axis limits
    all_chi = vcat([r.predicted_chi for r in results], [r.actual_chi for r in results])
    max_chi = maximum(all_chi)
    min_chi = max(1, minimum(all_chi))

    # Identity line
    lines!(ax, [min_chi, max_chi], [min_chi, max_chi],
           color = :gray, linestyle = :dash, linewidth = 1,
           label = "χ_pred = χ_actual")

    # Plot each circuit type
    for ct in circuit_types
        ct_results = filter(r -> r.circuit_type == ct, results)
        pred = [r.predicted_chi for r in ct_results]
        actual = [r.actual_chi for r in ct_results]
        color = get(COLORS, Symbol(ct), "#666666")

        scatter!(ax, pred, actual,
                 color = color,
                 markersize = 6,
                 label = ct)
    end

    # Add legend
    axislegend(ax, position = :rb)

    xlims!(ax, 0.5, max_chi * 1.1)
    ylims!(ax, 0.5, max_chi * 1.1)

    if save_path !== nothing
        save(save_path, fig, px_per_unit = DPI/72)
        println("Saved: $save_path")
    end

    fig
end

"""
    figure_ofd_vs_obd(results::Vector{StrategyComparisonResult}; save_path=nothing)

Generate Figure 2: OFD vs OBD comparison (2x2 panel).

(a) Wall-clock time vs T-count
(b) Wall-clock time vs system size
(c) Bond dimension comparison
(d) OFD success rate
"""
function figure_ofd_vs_obd(results::Vector{StrategyComparisonResult}; save_path=nothing)
    if !MAKIE_AVAILABLE
        println("CairoMakie not available. Generating text summary instead.")
        print_strategy_summary_text(results)
        return nothing
    end

    setup_makie_theme()

    fig = Figure(size = (700, 600))

    strategies = ["OFD", "OBD", "Hybrid"]

    # Panel (a): Time vs T-count
    ax_a = Axis(fig[1, 1],
        xlabel = "T-gate count",
        ylabel = "Time (s)",
        title = "(a) Time vs T-count",
        yscale = log10
    )

    t_counts = sort(unique(r.t_count for r in results))

    for strat in strategies
        times_by_t = Float64[]
        for t in t_counts
            t_data = [r.results[strat].wall_time_s for r in results
                      if r.t_count == t && haskey(r.results, strat)]
            if !isempty(t_data)
                push!(times_by_t, mean(t_data))
            else
                push!(times_by_t, NaN)
            end
        end

        valid = .!isnan.(times_by_t)
        color = COLORS[Symbol(lowercase(strat))]

        scatterlines!(ax_a, t_counts[valid], times_by_t[valid],
                      color = color, label = strat,
                      marker = MARKERS[Symbol(lowercase(strat))])
    end

    axislegend(ax_a, position = :lt)

    # Panel (b): Time vs system size
    ax_b = Axis(fig[1, 2],
        xlabel = "Number of qubits (n)",
        ylabel = "Time (s)",
        title = "(b) Time vs System Size",
        yscale = log10
    )

    n_qubits = sort(unique(r.n_qubits for r in results))

    for strat in strategies
        times_by_n = Float64[]
        for n in n_qubits
            n_data = [r.results[strat].wall_time_s for r in results
                      if r.n_qubits == n && haskey(r.results, strat)]
            if !isempty(n_data)
                push!(times_by_n, mean(n_data))
            else
                push!(times_by_n, NaN)
            end
        end

        valid = .!isnan.(times_by_n)
        color = COLORS[Symbol(lowercase(strat))]

        scatterlines!(ax_b, n_qubits[valid], times_by_n[valid],
                      color = color, label = strat,
                      marker = MARKERS[Symbol(lowercase(strat))])
    end

    # Panel (c): Bond dimension comparison
    ax_c = Axis(fig[2, 1],
        xlabel = "T-gate count",
        ylabel = "Bond dimension (χ)",
        title = "(c) Bond Dimension"
    )

    for strat in strategies
        chi_by_t = Float64[]
        for t in t_counts
            t_data = [r.results[strat].bond_dim for r in results
                      if r.t_count == t && haskey(r.results, strat)]
            if !isempty(t_data)
                push!(chi_by_t, mean(t_data))
            else
                push!(chi_by_t, NaN)
            end
        end

        valid = .!isnan.(chi_by_t)
        color = COLORS[Symbol(lowercase(strat))]

        scatterlines!(ax_c, t_counts[valid], chi_by_t[valid],
                      color = color, label = strat,
                      marker = MARKERS[Symbol(lowercase(strat))])
    end

    # Panel (d): OFD success rate
    ax_d = Axis(fig[2, 2],
        xlabel = "T-gate count",
        ylabel = "OFD Success Rate (%)",
        title = "(d) OFD Applicability"
    )

    for strat in ["OFD", "Hybrid"]
        rate_by_t = Float64[]
        for t in t_counts
            t_data = [r.results[strat].ofd_success_rate for r in results
                      if r.t_count == t && haskey(r.results, strat)]
            if !isempty(t_data)
                push!(rate_by_t, 100 * mean(t_data))
            else
                push!(rate_by_t, NaN)
            end
        end

        valid = .!isnan.(rate_by_t)
        color = COLORS[Symbol(lowercase(strat))]

        scatterlines!(ax_d, t_counts[valid], rate_by_t[valid],
                      color = color, label = strat,
                      marker = MARKERS[Symbol(lowercase(strat))])
    end

    ylims!(ax_d, 0, 105)
    axislegend(ax_d, position = :rb)

    if save_path !== nothing
        save(save_path, fig, px_per_unit = DPI/72)
        println("Saved: $save_path")
    end

    fig
end

"""
    figure_scaling_analysis(results::Vector{ScalingDataPoint}; save_path=nothing)

Generate Figure 3: Scaling analysis (2x2 panel).

(a) Clifford-only scaling
(b) Clifford+T with OFD scaling
(c) Bond dimension vs T-count
(d) With vs without disentangling
"""
function figure_scaling_analysis(results::Vector{ScalingDataPoint}; save_path=nothing)
    if !MAKIE_AVAILABLE
        println("CairoMakie not available. Generating text summary instead.")
        print_scaling_summary_text(results)
        return nothing
    end

    setup_makie_theme()

    fig = Figure(size = (700, 600))

    # Panel (a): Clifford-only scaling
    ax_a = Axis(fig[1, 1],
        xlabel = "Number of qubits (n)",
        ylabel = "Time (s)",
        title = "(a) Clifford-only Scaling",
        xscale = log10,
        yscale = log10
    )

    clifford_data = filter(r -> r.parameter_name == "n_clifford_only", results)
    if !isempty(clifford_data)
        x = [r.parameter_value for r in clifford_data]
        y = [r.mean_time for r in clifford_data]
        yerr = [r.std_time for r in clifford_data]

        errorbars!(ax_a, x, y, yerr, color = COLORS[:ofd], whiskerwidth = 5)
        scatter!(ax_a, x, y, color = COLORS[:ofd], markersize = 8)

        # Fit and plot power law
        a, b = fit_power_law_simple(x, y)
        if !isnan(b)
            x_fit = range(minimum(x), maximum(x), length=50)
            y_fit = a .* x_fit.^b
            lines!(ax_a, collect(x_fit), y_fit,
                   color = :gray, linestyle = :dash,
                   label = @sprintf("∝ n^{%.1f}", b))
            axislegend(ax_a, position = :lt)
        end
    end

    # Panel (b): Clifford+T OFD scaling
    ax_b = Axis(fig[1, 2],
        xlabel = "Number of qubits (n)",
        ylabel = "Time (s)",
        title = "(b) Clifford+T with OFD",
        xscale = log10,
        yscale = log10
    )

    ofd_data = filter(r -> r.parameter_name == "n_ofd", results)
    if !isempty(ofd_data)
        x = [r.parameter_value for r in ofd_data]
        y = [r.mean_time for r in ofd_data]
        yerr = [r.std_time for r in ofd_data]

        errorbars!(ax_b, x, y, yerr, color = COLORS[:ofd], whiskerwidth = 5)
        scatter!(ax_b, x, y, color = COLORS[:ofd], markersize = 8)

        a, b = fit_power_law_simple(x, y)
        if !isnan(b)
            x_fit = range(minimum(x), maximum(x), length=50)
            y_fit = a .* x_fit.^b
            lines!(ax_b, collect(x_fit), y_fit,
                   color = :gray, linestyle = :dash,
                   label = @sprintf("∝ n^{%.1f}", b))
            axislegend(ax_b, position = :lt)
        end
    end

    # Panel (c): Bond dimension patterns
    ax_c = Axis(fig[2, 1],
        xlabel = "T-gate count",
        ylabel = "Bond dimension (χ)",
        title = "(c) Bond Dimension Growth"
    )

    indep_data = filter(r -> r.parameter_name == "chi_independent", results)
    entangled_data = filter(r -> r.parameter_name == "chi_entangled", results)

    if !isempty(indep_data)
        x = [r.parameter_value for r in indep_data]
        y = [r.mean_bond_dim for r in indep_data]
        scatterlines!(ax_c, x, y, color = COLORS[:independent],
                      label = "Independent", marker = :circle)
    end

    if !isempty(entangled_data)
        x = [r.parameter_value for r in entangled_data]
        y = [r.mean_bond_dim for r in entangled_data]
        scatterlines!(ax_c, x, y, color = COLORS[:entangled],
                      label = "Entangled", marker = :rect)
    end

    axislegend(ax_c, position = :lt)

    # Panel (d): T-count scaling comparison
    ax_d = Axis(fig[2, 2],
        xlabel = "T-gate count",
        ylabel = "Bond dimension (χ)",
        title = "(d) With vs Without Disentangling",
        yscale = log10
    )

    ofd_t_data = filter(r -> r.parameter_name == "t_ofd", results)
    none_t_data = filter(r -> r.parameter_name == "t_none", results)

    if !isempty(ofd_t_data)
        x = [r.parameter_value for r in ofd_t_data]
        y = [r.mean_bond_dim for r in ofd_t_data]
        scatterlines!(ax_d, x, y, color = COLORS[:ofd],
                      label = "With OFD", marker = :circle)
    end

    if !isempty(none_t_data)
        x = [r.parameter_value for r in none_t_data]
        y = [r.mean_bond_dim for r in none_t_data]
        scatterlines!(ax_d, x, y, color = COLORS[:none],
                      label = "No disentangling", marker = :rect)

        # Add 2^t reference line
        x_ref = 1:maximum(x)
        y_ref = 2.0 .^ x_ref
        lines!(ax_d, collect(x_ref), y_ref,
               color = :gray, linestyle = :dash, label = "2^t")
    end

    axislegend(ax_d, position = :lt)

    if save_path !== nothing
        save(save_path, fig, px_per_unit = DPI/72)
        println("Saved: $save_path")
    end

    fig
end

"""
    figure_hybrid_benefits(results::Vector{StrategyComparisonResult}; save_path=nothing)

Generate Figure 4: Hybrid strategy benefits across circuit types.
"""
function figure_hybrid_benefits(results::Vector{StrategyComparisonResult}; save_path=nothing)
    if !MAKIE_AVAILABLE
        println("CairoMakie not available. Generating text summary instead.")
        print_hybrid_summary_text(results)
        return nothing
    end

    setup_makie_theme()

    fig = Figure(size = (600, 400))

    circuit_types = unique(r.circuit_name for r in results)
    strategies = ["OFD", "OBD", "Hybrid"]

    # Bar plot comparing times across circuit types
    ax = Axis(fig[1, 1],
        xlabel = "Circuit Type",
        ylabel = "Mean Time (s)",
        title = "Strategy Comparison Across Circuit Types",
        xticks = (1:length(circuit_types), circuit_types)
    )

    width = 0.25
    offsets = [-width, 0, width]

    for (i, strat) in enumerate(strategies)
        times = Float64[]
        for ct in circuit_types
            ct_data = [r.results[strat].wall_time_s for r in results
                       if r.circuit_name == ct && haskey(r.results, strat)]
            push!(times, isempty(ct_data) ? 0.0 : mean(ct_data))
        end

        color = COLORS[Symbol(lowercase(strat))]
        barplot!(ax, (1:length(circuit_types)) .+ offsets[i], times,
                 width = width, color = color, label = strat)
    end

    axislegend(ax, position = :rt)

    if save_path !== nothing
        save(save_path, fig, px_per_unit = DPI/72)
        println("Saved: $save_path")
    end

    fig
end

"""
    generate_all_figures(suite::BenchmarkSuite, output_dir::String)

Generate all publication figures from a benchmark suite.
"""
function generate_all_figures(suite::BenchmarkSuite, output_dir::String)
    mkpath(output_dir)

    println("Generating publication figures...")
    println("Output directory: $output_dir")

    # Figure 1: GF(2) validation
    if haskey(suite.experiments, "gf2_validation")
        results = suite.experiments["gf2_validation"].results
        if !isempty(results)
            fig1_path = joinpath(output_dir, "fig1_gf2_validation.pdf")
            try
                figure_gf2_validation(results; save_path=fig1_path)
            catch e
                @warn "Failed to generate GF(2) validation figure" exception=e
                print_gf2_summary(results)
            end
        end
    end

    # Figure 2: OFD vs OBD
    if haskey(suite.experiments, "ofd_vs_obd")
        results = suite.experiments["ofd_vs_obd"].results
        if !isempty(results)
            fig2_path = joinpath(output_dir, "fig2_ofd_vs_obd.pdf")
            try
                figure_ofd_vs_obd(results; save_path=fig2_path)
            catch e
                @warn "Failed to generate OFD vs OBD figure" exception=e
                print_strategy_summary_text(results)
            end
        end
    end

    # Figure 3: Scaling
    if haskey(suite.experiments, "scaling")
        results = suite.experiments["scaling"].results
        if !isempty(results)
            fig3_path = joinpath(output_dir, "fig3_scaling.pdf")
            try
                figure_scaling_analysis(results; save_path=fig3_path)
            catch e
                @warn "Failed to generate scaling figure" exception=e
                print_scaling_summary_text(results)
            end
        end
    end

    # Figure 4: Hybrid benefits
    if haskey(suite.experiments, "hybrid_strategy")
        results = suite.experiments["hybrid_strategy"].results
        if !isempty(results)
            fig4_path = joinpath(output_dir, "fig4_hybrid_benefits.pdf")
            try
                figure_hybrid_benefits(results; save_path=fig4_path)
            catch e
                @warn "Failed to generate hybrid benefits figure" exception=e
                print_hybrid_summary_text(results)
            end
        end
    end

    println("Figure generation complete!")
end

# Helper functions

"""
    fit_power_law_simple(x, y) -> (a, b)

Fit y = a * x^b using log-log linear regression.
"""
function fit_power_law_simple(x::Vector, y::Vector)
    valid = (x .> 0) .& (y .> 0) .& isfinite.(x) .& isfinite.(y)

    if sum(valid) < 2
        return (NaN, NaN)
    end

    log_x = log.(x[valid])
    log_y = log.(y[valid])

    x_mean = mean(log_x)
    y_mean = mean(log_y)

    num = sum((log_x .- x_mean) .* (log_y .- y_mean))
    den = sum((log_x .- x_mean).^2)

    b = den > 0 ? num / den : NaN
    a = exp(y_mean - b * x_mean)

    (a, b)
end

# Text-based fallbacks when Makie is not available

function print_gf2_summary(results::Vector{GF2ValidationResult})
    println("\nGF(2) Validation Summary (text mode):")
    println("-"^50)

    n_total = length(results)
    n_exact = count(r -> r.exact_match, results)

    println("Total tests: $n_total")
    println("Exact matches: $n_exact ($(round(100*n_exact/n_total, digits=1))%)")

    circuit_types = unique(r.circuit_type for r in results)
    for ct in circuit_types
        ct_results = filter(r -> r.circuit_type == ct, results)
        n_ct_exact = count(r -> r.exact_match, ct_results)
        @printf("  %-20s: %d/%d exact\n", ct, n_ct_exact, length(ct_results))
    end
end

function print_strategy_summary_text(results::Vector{StrategyComparisonResult})
    println("\nStrategy Comparison Summary (text mode):")
    println("-"^50)

    strategies = collect(keys(first(results).results))
    for strat in strategies
        times = [r.results[strat].wall_time_s for r in results if haskey(r.results, strat)]
        @printf("  %-8s: mean time = %.4fs\n", strat, mean(times))
    end
end

function print_scaling_summary_text(results::Vector{ScalingDataPoint})
    println("\nScaling Analysis Summary (text mode):")
    println("-"^50)

    param_names = unique(r.parameter_name for r in results)
    for pname in param_names
        pdata = filter(r -> r.parameter_name == pname, results)
        println("  $pname: $(length(pdata)) data points")
    end
end

function print_hybrid_summary_text(results::Vector{StrategyComparisonResult})
    println("\nHybrid Strategy Summary (text mode):")
    println("-"^50)

    circuit_types = unique(r.circuit_name for r in results)
    for ct in circuit_types
        println("  $ct: $(count(r -> r.circuit_name == ct, results)) tests")
    end
end

export setup_makie_theme
export figure_gf2_validation, figure_ofd_vs_obd
export figure_scaling_analysis, figure_hybrid_benefits
export generate_all_figures
export COLORS, MARKERS
