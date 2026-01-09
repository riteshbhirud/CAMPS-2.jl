#!/usr/bin/env julia
# Examine the failed Phase 2 circuits from the results file

using CSV
using DataFrames

println("="^80)
println("ANALYZING FAILED CIRCUITS")
println("="^80)
println()

# Load results
results_file = "results/complete_benchmark/results_20260108_122139.csv"
println("Loading: $results_file")

df = CSV.read(results_file, DataFrame)

println("Total circuits: ", nrow(df))
println()

# Filter failed circuits
failed = filter(row -> row.success == false, df)
println("Failed circuits: ", nrow(failed))
println()

if nrow(failed) > 0
    # Group by family
    println("Failed circuits by family:")
    println("-"^80)
    
    failed_by_family = combine(groupby(failed, :family), nrow => :count)
    sort!(failed_by_family, :count, rev=true)
    
    for row in eachrow(failed_by_family)
        println("  ", rpad(row.family, 40), " : ", row.count, " failures")
    end
    
    println()
    println("="^80)
    println("SAMPLE ERROR MESSAGES (First 5 failed circuits):")
    println("="^80)
    println()
    
    for (i, row) in enumerate(eachrow(failed[1:min(5, nrow(failed)), :]))
        println("[$i] Family: ", row.family)
        println("    Params: n_qubits=", row.n_qubits, ", seed=", row.seed)
        
        if hasproperty(row, :error) && !ismissing(row.error)
            println("    Error: ", first(row.error, 500), "...")
        else
            println("    Error: [No error message recorded]")
        end
        println()
    end
    
    # Check if errors are similar
    println("="^80)
    println("ERROR PATTERN ANALYSIS:")
    println("="^80)
    println()
    
    if hasproperty(failed, :error)
        # Extract first line of each error
        error_types = Dict{String, Int}()
        
        for row in eachrow(failed)
            if !ismissing(row.error)
                # Get first line of error
                first_line = split(string(row.error), "\n")[1]
                # Truncate to first 100 chars
                key = first(first_line, 100)
                error_types[key] = get(error_types, key, 0) + 1
            end
        end
        
        println("Common error patterns:")
        sorted_errors = sort(collect(error_types), by=x->x[2], rev=true)
        for (i, (err, count)) in enumerate(sorted_errors[1:min(5, length(sorted_errors))])
            println("  [$i] ($count occurrences)")
            println("       ", err)
            println()
        end
    end
else
    println("No failed circuits found!")
end