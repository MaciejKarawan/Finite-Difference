# Finite Difference Options Pricing with Uncertain Volatility

## Project Overview

This project implements explicit finite difference methods for pricing complex derivative instruments with uncertain volatility, focusing on WIG20 index-based options. The analysis explores the impact of volatility uncertainty on option pricing and optimal exercise strategies.

## Features

### Option Types Analyzed
- **European Call/Put Knock-out Options**
- **American Call/Put Knock-out Options**

### Key Parameters
- **Underlying**: WIG20 Index
- **Maturity**: December 31, 2025
- **Strike Price**: 2,200
- **Barriers**: 2,400 (call options), 1,900 (put options)
- **Volatility Range**: 20% - 30% (uncertain volatility)
- **Dividend**: 200 (fixed amount at mid-life)

## Technical Implementation

### Numerical Methods
- **Explicit Finite Difference Scheme** for PDE solving
- **Uncertain Volatility Modeling** with dynamic volatility bounds
- **Barrier Option Mechanics** with knock-out conditions
- **American Exercise Optimization** for early exercise decisions

### Interactive Visualizations
The project features comprehensive **interactive charts built with Plotly.go architecture**, providing:

- **3D Surface Plots** showing option prices vs. underlying price and time
- **Sensitivity Analysis Charts** (Greeks visualization)
- **Volatility Impact Comparison** between fixed and uncertain volatility scenarios  
- **Early Exercise Boundaries** for American options
- **Barrier Proximity Analysis** with knockout probability heatmaps
- **Dividend Impact Visualization** across different dividend levels

## Analysis Scope

### Parameter Sensitivity Studies
1. **Volatility Uncertainty Impact**
   - Comparison between fixed volatility (25%) and uncertain volatility (20%-30%)
   - Analysis of different volatility ranges
   - Worst-case scenario pricing for option sellers

2. **Dividend Effects**
   - Fixed dividend amount variations (x = 50, 100, 200, 400)
   - Comparison with percentage-based dividend scenarios
   - Timing impact on option values

3. **Barrier Analysis**
   - Distance to barrier effects on pricing
   - Knockout probability evolution
   - Barrier level sensitivity

4. **American vs European Comparison**
   - Early exercise premiums
   - Optimal exercise boundaries
   - Time decay analysis

## Validation & Testing

The implementation includes comprehensive validation through:
- **Boundary Condition Verification**
- **Comparison with Black-Scholes** for simplified cases
- **Monte Carlo Cross-validation** for complex scenarios
- **Convergence Testing** for numerical stability

## Key Findings

- Uncertain volatility significantly impacts option pricing, with effects varying by moneyness and time to expiration
- American options show complex early exercise patterns near barriers
- Dividend timing creates notable price discontinuities
- Interactive visualizations reveal non-linear relationships between parameters

## Technical Stack

- **Core Algorithm**: Explicit Finite Difference (Custom Implementation)
- **Visualization**: Plotly.go for interactive 3D charts
- **Data Processing**: Optimized numerical routines
- **Validation**: Multiple cross-validation approaches


## Usage

The project provides both computational engines and interactive visualization tools for:
- **Quantitative Analysts** researching volatility uncertainty effects
- **Risk Managers** evaluating worst-case pricing scenarios  
- **Options Traders** understanding complex parameter interactions
- **Academic Researchers** studying numerical methods in finance

## Results Highlights

The interactive charts demonstrate complex, non-linear relationships between:
- Option prices and volatility uncertainty bounds
- Early exercise timing and market conditions
- Barrier proximity and knockout risk
- Dividend policy and option valuations

All visualizations are accessible through web browsers with full interactivity for parameter exploration.

---

*This project was developed as part of Financial Engineering I course (2024/2025), demonstrating advanced numerical methods and interactive data visualization in quantitative finance.*
