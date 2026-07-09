-- heavy-search.lua
--
-- AWARE-style heavy-search phase for the DSB Hotel Reservation Phase R
-- shift-oscillation experiments. Not part of upstream DSB.
--
-- Difference from mixed-workload_type_1.lua:
--   * 100 % /hotels requests (drops recommend/user/reserve).
--     recommend/user/reserve exercise the 6th–10th services which
--     Phase R does not deploy, so they would 4xx.
--   * Widened lat/lon window (~10x upstream span) so the search →
--     geo/rate/profile fan-out has to touch more hotels per query,
--     saturating the leaf services.
--
-- Used by the Phase Q.5-style DSB_SHIFT_PHASES machinery to swap
-- workload mid-trial and reproduce the AWARE input-shift regime.

local socket = require("socket")
math.randomseed(socket.gettime()*1000)
math.random(); math.random(); math.random()

-- Empty prefix → wrk.format uses the target URL passed on the wrk2 command
-- line (http://frontend:5000) as request base. See mixed-workload_type_1.lua
-- for the full rationale.
local url = ""

local function search_hotel_heavy()
  local in_date  = math.random(9, 23)
  local out_date = math.random(in_date + 1, 24)

  local in_date_str  = string.format("2015-04-%02d", in_date)
  local out_date_str = string.format("2015-04-%02d", out_date)

  -- Widened window: upstream uses ±0.24 lat / ±0.16 lon around Napa.
  -- We use ±2.40 lat / ±1.60 lon (10 x wider), forcing search to
  -- fan out to a much larger candidate hotel set.
  local lat = 38.0235 + (math.random(0, 4810) - 2405) / 1000.0
  local lon = -122.095 + (math.random(0, 3250) - 1570) / 1000.0

  local method  = "GET"
  local path    = url .. "/hotels?inDate=" .. in_date_str ..
                  "&outDate=" .. out_date_str ..
                  "&lat=" .. tostring(lat) ..
                  "&lon=" .. tostring(lon)
  local headers = {}
  return wrk.format(method, path, headers, nil)
end

request = function()
  return search_hotel_heavy()
end
