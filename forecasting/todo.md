**Things to check/fix in the dataset**

1. Time alignment (most important)
* Compute the cross‑correlation between power and irradiance/humidity to see if correlation peaks at a non‑zero lag. If it does, shift the weather covariates accordingly.
* Quick check: For daytime only (irradiance > 50 W/m²), plot power vs irradiance. If still very diffuse, you likely have a lag or quality issue.

2. Circular wind direction
* Don’t feed raw degrees. Convert to sin/cos or to wind components using velocity:
u = wind_velocity * cos(dir_rad)
v = wind_velocity * sin(dir_rad)
* Linear correlation on raw degrees is misleading.

3. Capacity normalization and outliers
* If you know plant capacity, normalize power to [0,1] (capacity factor).


**What additional diagrams would be most useful**

1. Lag/lead analysis:
* Cross‑correlation of power with irradiance (and humidity) across ±6 hours.
* Scatter of power vs irradiance colored by hour‑of‑day (or binned by hour).

2. Day/season structure:
* Boxplots (or violin) of power by hour‑of‑day and by month.
* STL decomposition or seasonal plot to visualize daily/annual seasonality.

3. Conditional relationships:
* Power vs Irradiance hexbin/2D density, colored by temperature bin (low/med/high). Expect lower power at high temp for the same irradiance if alignment is correct.

4. Residual diagnostics:
* Fit a simple daytime model (e.g., power ~ irradiance) and plot residuals vs temperature, humidity, wind. This reveals secondary effects better than raw correlations.

**Feature engineering I’d add (most will be “known” into the future)**

1. Time/solar geometry (deterministic, strong signal)
*Hour‑of‑day and day‑of‑year as sin/cos pairs.
*Solar elevation/azimuth and “sun_up” indicator (solar elevation > 0).

2. Clear‑sky and cloudiness
* Compute clear‑sky GHI with pvlib, then clear‑sky index k_t = measured_irradiance / clearsky_irradiance. k_t is a strong proxy for clouds and generalizes well.

3. Module/cell temperature and thermal effects
* T_cell ≈ T_air + (NOCT − 20)/800 * irradiance.
* Interaction: T_cell × irradiance, or include expected efficiency term 1 + γ·(T_cell−25°C).

4. Humidity‑related
* Dew point and vapor pressure deficit (VPD). These often track cloud formation/fog better than RH alone.

5. Wind
* Use u,v components instead of direction; rolling mean of wind (cooling effect on modules).

6. Precipitation/rain
* Binary rain flag and short lags (rain at t−1…t−3). Helps capture sudden irradiance drops and cleaning effects after rain.

7. Lags and rolling stats (use only if you can provide them at inference or restrict to target-only lags)
* Target lags: power_{t−1, t−2, t−24}, plus rolling mean/std over 3h/6h/24h.
* Exogenous lags: irradiance_{t−1}, humidity_{t−1} if your deployment can supply forecasted covariates for the horizon.
