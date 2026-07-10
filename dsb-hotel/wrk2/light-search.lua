-- light-search.lua
--
-- Phase R.5 companion to heavy-search.lua for the oscillating regime
-- that specifically breaks VPA's histogram-based recommender (see
-- AWARE, USENIX ATC '23 §5). Not part of upstream DSB.
--
-- Difference from heavy-search.lua:
--   * NARROW lat/lon window (±0.05° from Napa centre — ~5 km radius)
--     instead of heavy's ±2.40° (~250 km). The narrow window keeps
--     mongodb-geo returning the same handful of hotels every call,
--     so memcached-geo cache-hits ~100 % of queries after the first
--     couple of seconds. Per-request work drops from mongo scan +
--     rate compute to a memcached lookup — CPU on frontend, search,
--     and profile falls to ~20 % of request.
--
-- Used by the oscillating DSB_SHIFT_PHASES pattern:
--     DSB_SHIFT_PHASES=heavy-search.lua:60,light-search.lua:60,…
-- so every 60 s the leaves flip between saturating memcached-warm
-- and full mongo-cold traffic. VPA's histogram averages the two into
-- a mid-value that fits neither regime; AutoSage's per-tick LLM sees
-- the current regime directly and sizes for it.

local socket = require("socket")
math.randomseed(socket.gettime()*1000)
math.random(); math.random(); math.random()

local url = ""  -- see mixed-workload_type_1.lua for rationale

local function search_hotel_light()
  local in_date  = math.random(9, 23)
  local out_date = math.random(in_date + 1, 24)

  local in_date_str  = string.format("2015-04-%02d", in_date)
  local out_date_str = string.format("2015-04-%02d", out_date)

  -- Narrow window: ±0.05° lat / ±0.05° lon (roughly ±5 km) around
  -- the same Napa centre heavy-search uses. Only ~10 unique
  -- (lat, lon) buckets get exercised, so mongodb-geo returns the
  -- same result set every time and memcached-geo caches everything
  -- after one iteration. Per-request cost drops to a memcached hit.
  local lat = 38.0235 + (math.random(0, 100) - 50) / 1000.0
  local lon = -122.095 + (math.random(0, 100) - 50) / 1000.0

  local method  = "GET"
  local path    = url .. "/hotels?inDate=" .. in_date_str ..
                  "&outDate=" .. out_date_str ..
                  "&lat=" .. tostring(lat) ..
                  "&lon=" .. tostring(lon)
  local headers = {}
  return wrk.format(method, path, headers, nil)
end

request = function()
  return search_hotel_light()
end
