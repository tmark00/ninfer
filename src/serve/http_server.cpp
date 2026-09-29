#include "serve/http_server.h"
#include <spdlog/spdlog.h>
#include <fstream>
#include "serve/mcp_proxy.h"
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

#include "serve/anthropic_messages.h"
#include "serve/http_transport.h"
#include "serve/openai_common.h"
#include "serve/request_log.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::serve {
namespace {

void write_exception(httplib::Response& res, const std::exception& ex) {
    ApiError error;
    error.status  = 500;
    error.type    = "internal_error";
    error.message = ex.what();
    write_openai_error(res, error);
}

bool is_anthropic_path(std::string_view path) { return path.starts_with("/v1/messages"); }

bool is_openai_path(std::string_view path) {
    return path.starts_with("/v1/") && !is_anthropic_path(path);
}

void ensure_openai_request_id(const httplib::Request& request, httplib::Response& response) {
    if (is_openai_path(request.path) && !response.has_header("x-request-id")) {
        response.set_header("x-request-id", new_openai_request_id());
    }
}

ThroughputReport make_throughput_report(const ninfer::RuntimeStats& previous,
                                        const ninfer::RuntimeStats& current,
                                        double interval_seconds) {
    return ThroughputReport{
        .interval_seconds = interval_seconds,
        .computed_prefill_tokens =
            current.computed_prefill_tokens - previous.computed_prefill_tokens,
        .committed_decode_tokens =
            current.committed_decode_tokens - previous.committed_decode_tokens,
        .decode_rounds     = current.decode_rounds - previous.decode_rounds,
        .decode_row_rounds = current.decode_row_rounds - previous.decode_row_rounds,
        .previous          = previous,
        .current           = current,
    };
}

bool report_has_activity(const ThroughputReport& report) {
    return report.computed_prefill_tokens != 0 || report.committed_decode_tokens != 0 ||
           report.decode_rounds != 0 || report.current.running_requests != 0 ||
           report.current.waiting_requests != 0 || report.current.materializing_requests != 0 ||
           report.current.capture_pending_requests != 0 ||
           report.current.terminal_pending_requests != 0 ||
           report.current.active_captures_completed != report.previous.active_captures_completed ||
           report.current.active_captures_aborted != report.previous.active_captures_aborted ||
           report.current.root_selections != report.previous.root_selections ||
           report.current.private_endpoint_selections !=
               report.previous.private_endpoint_selections ||
           report.current.private_turn_closure_selections !=
               report.previous.private_turn_closure_selections ||
           report.current.private_response_replay_selections !=
               report.previous.private_response_replay_selections ||
           report.current.private_long_anchor_selections !=
               report.previous.private_long_anchor_selections ||
           report.current.shared_stable_prefix_selections !=
               report.previous.shared_stable_prefix_selections ||
           report.current.state_moves != report.previous.state_moves ||
           report.current.state_forks != report.previous.state_forks ||
           report.current.state_restores != report.previous.state_restores ||
           report.current.state_d2h_count != report.previous.state_d2h_count ||
           report.current.state_h2d_count != report.previous.state_h2d_count ||
           report.current.state_d2d_count != report.previous.state_d2d_count ||
           report.current.main_kv_d2h_pages != report.previous.main_kv_d2h_pages ||
           report.current.main_kv_h2d_pages != report.previous.main_kv_h2d_pages ||
           report.current.main_kv_d2d_pages != report.previous.main_kv_d2d_pages ||
           report.current.backend_kv_d2h_pages != report.previous.backend_kv_d2h_pages ||
           report.current.backend_kv_h2d_pages != report.previous.backend_kv_h2d_pages ||
           report.current.backend_kv_d2d_pages != report.previous.backend_kv_d2d_pages ||
           report.current.pressure_spill_pages != report.previous.pressure_spill_pages ||
           report.current.partial_tail_cow_pages != report.previous.partial_tail_cow_pages ||
           report.current.pressure_private_owners_degraded !=
               report.previous.pressure_private_owners_degraded ||
           report.current.pressure_private_owners_evicted !=
               report.previous.pressure_private_owners_evicted ||
           report.current.pressure_shared_owners_degraded !=
               report.previous.pressure_shared_owners_degraded ||
           report.current.pressure_shared_owners_evicted !=
               report.previous.pressure_shared_owners_evicted ||
           report.current.pressure_checkpoints_dropped !=
               report.previous.pressure_checkpoints_dropped ||
           report.current.pressure_searches != report.previous.pressure_searches ||
           report.current.pressure_search_budget_exhaustions !=
               report.previous.pressure_search_budget_exhaustions ||
           report.current.pressure_maximal_fallback_selections !=
               report.previous.pressure_maximal_fallback_selections ||
           report.current.historical_fork_hits != report.previous.historical_fork_hits ||
           report.current.device_state_occupied_slots !=
               report.previous.device_state_occupied_slots ||
           report.current.host_state_occupied_slots != report.previous.host_state_occupied_slots ||
           report.current.device_main_kv_occupied_pages !=
               report.previous.device_main_kv_occupied_pages ||
           report.current.device_backend_kv_occupied_pages !=
               report.previous.device_backend_kv_occupied_pages ||
           report.current.host_kv_occupied_bytes != report.previous.host_kv_occupied_bytes ||
           report.current.shared_active_references != report.previous.shared_active_references ||
           report.current.host_work.engine_boundary_ns !=
               report.previous.host_work.engine_boundary_ns ||
           report.current.host_work.program_submit_ns !=
               report.previous.host_work.program_submit_ns ||
           report.current.host_work.program_post_ns != report.previous.host_work.program_post_ns ||
           report.current.host_work.engine_commit_output_ns !=
               report.previous.host_work.engine_commit_output_ns ||
           report.current.host_work.engine_maintenance_ns !=
               report.previous.host_work.engine_maintenance_ns ||
           report.current.host_work.device_wait_ns != report.previous.host_work.device_wait_ns;
}

const char* endpoint_name(std::string_view path) noexcept {
    if (path == "/v1/chat/completions") { return "openai_chat_completions"; }
    if (path == "/v1/responses") { return "openai_responses"; }
    if (path == "/v1/responses/input_tokens") { return "openai_responses_input_tokens"; }
    if (path == "/v1/messages") { return "anthropic_messages"; }
    if (path == "/v1/messages/count_tokens") { return "anthropic_count_tokens"; }
    return "http_route";
}

std::string response_request_id(const httplib::Response& response) {
    if (response.has_header("x-request-id")) { return response.get_header_value("x-request-id"); }
    if (response.has_header("request-id")) { return response.get_header_value("request-id"); }
    return {};
}

} // namespace

void write_openai_error(httplib::Response& response, const ApiError& error) {
    response.status = error.status;
    response.set_content(make_error_body(error), "application/json");
}

void write_anthropic_error(httplib::Response& response, const ApiError& api_error,
                           const std::string& request_id) {
    const ApiError error = normalize_anthropic_error(api_error);
    response.status      = error.status;
    response.headers.erase("request-id");
    response.set_header("request-id", request_id);
    response.set_content(make_anthropic_error_body(error, request_id), "application/json");
}

httplib::Server::HandlerResponse handle_unrendered_http_error(const ServeOptions& options,
                                                              const httplib::Request& request,
                                                              httplib::Response& response) {
    ensure_openai_request_id(request, response);
    if (!response.body.empty()) { return httplib::Server::HandlerResponse::Unhandled; }

    ApiError error;
    if (response.status == 413) {
        error.status  = 413;
        error.type    = "invalid_request_error";
        error.code    = "request_too_large";
        error.message = "request body exceeds the configured payload limit of " +
                        std::to_string(options.max_request_bytes) + " bytes";
    } else if (response.status == 404 && request.path.rfind("/v1/messages", 0) == 0) {
        error.status  = 404;
        error.code    = "not_found";
        error.message = "requested Anthropic resource was not found";
    } else {
        return httplib::Server::HandlerResponse::Unhandled;
    }
    if (request.path.rfind("/v1/messages", 0) == 0) {
        write_anthropic_error(response, error, new_anthropic_request_id());
    } else {
        write_openai_error(response, error);
    }
    return httplib::Server::HandlerResponse::Handled;
}

bool matches_bearer_credential(std::string_view authorization, std::string_view api_key) noexcept {
    if (api_key.empty()) { return false; }
    const auto is_whitespace = [](char value) { return value == ' ' || value == '\t'; };
    const auto ascii_equal   = [](char lhs, char rhs) {
        if (lhs >= 'A' && lhs <= 'Z') { lhs = static_cast<char>(lhs - 'A' + 'a'); }
        if (rhs >= 'A' && rhs <= 'Z') { rhs = static_cast<char>(rhs - 'A' + 'a'); }
        return lhs == rhs;
    };

    std::size_t position = 0;
    while (position < authorization.size() && is_whitespace(authorization[position])) {
        ++position;
    }
    constexpr std::string_view scheme = "Bearer";
    if (authorization.size() - position < scheme.size()) { return false; }
    for (std::size_t index = 0; index < scheme.size(); ++index) {
        if (!ascii_equal(authorization[position + index], scheme[index])) { return false; }
    }
    position += scheme.size();
    if (position == authorization.size() || !is_whitespace(authorization[position])) {
        return false;
    }
    while (position < authorization.size() && is_whitespace(authorization[position])) {
        ++position;
    }
    std::size_t end = authorization.size();
    while (end > position && is_whitespace(authorization[end - 1])) { --end; }
    return authorization.substr(position, end - position) == api_key;
}

HttpServer::HttpServer(ServeOptions options, std::shared_ptr<spdlog::logger> logger)
    : options_(std::move(options)), openai_responses_store_(options_.response_store_max_records,
                                                            options_.response_store_max_bytes),
      operational_log_(logger),
      request_jsonl_(options_.request_log_jsonl, options_.artifact_path, std::move(logger)) {
    const std::size_t queued_requests =
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests;
    const std::size_t worker_count = queued_requests + 1;
    server_.new_task_queue         = [queued_requests, worker_count] {
        return new httplib::ThreadPool(worker_count, worker_count, queued_requests);
    };
    server_.set_socket_options(configure_http_server_socket);
    server_.set_payload_max_length(options_.max_request_bytes);
    if (!options_.webui_dir.empty()) {
        mount_webui(options_.webui_dir);
        register_webui_mime();
    }
    register_routes();
}

HttpServer::RequestLifecycle::RequestLifecycle(HttpServer& owner, RequestLogContext context)
    : owner_(&owner), context_(std::move(context)) {
    owner_->record_request_start(context_);
}

bool HttpServer::RequestLifecycle::claim(State terminal) noexcept {
    State expected = State::Pending;
    return state_.compare_exchange_strong(expected, terminal, std::memory_order_acq_rel);
}

void HttpServer::RequestLifecycle::done(const GenerationOutcome& outcome) {
    if (claim(State::Done)) { owner_->record_request_done(context_, outcome); }
}

void HttpServer::RequestLifecycle::failure(const RequestFailure& failure) {
    if (claim(State::Error)) { owner_->record_request_failure(context_, failure); }
}

void HttpServer::RequestLifecycle::response_failure(const RequestFailure& failure) {
    owner_->record_response_failure(context_.id, failure);
}

std::shared_ptr<HttpServer::RequestLifecycle> HttpServer::begin_request(RequestLogContext context) {
    return std::make_shared<RequestLifecycle>(*this, std::move(context));
}

void HttpServer::record_request_start(const RequestLogContext& context) {
    request_jsonl_.write_request_start(context);
    operational_log_.request_start(context);
}

void HttpServer::record_request_rejected(const RequestRejectionLogContext& context) {
    request_jsonl_.write_request_rejected(context);
    operational_log_.request_rejected(context);
}

void HttpServer::record_request_done(const RequestLogContext& context,
                                     const GenerationOutcome& outcome) {
    request_jsonl_.write_request_done(context, outcome);
    operational_log_.request_done(context, outcome);
}

void HttpServer::record_request_failure(const RequestLogContext& context,
                                        const RequestFailure& failure) {
    request_jsonl_.write_request_error(context, failure.machine_message);
    operational_log_.request_failure(context, failure);
}

void HttpServer::record_response_failure(std::uint64_t request_id, const RequestFailure& failure) {
    operational_log_.response_failure(request_id, failure);
}

void HttpServer::record_throughput(const ThroughputReport& report) {
    request_jsonl_.write_throughput(report);
    operational_log_.throughput(report);
}

void HttpServer::run_stats_reporter() {
    using Clock                     = std::chrono::steady_clock;
    ninfer::RuntimeStats previous   = service_->runtime_stats();
    Clock::time_point previous_time = Clock::now();
    const auto interval             = std::chrono::milliseconds(options_.log_stats_interval_ms);
    Clock::time_point next_deadline = previous_time + interval;

    for (;;) {
        {
            std::unique_lock lock(stats_mutex_);
            if (stats_cv_.wait_until(lock, next_deadline, [this] { return stats_stopping_; })) {
                break;
            }
        }

        const ninfer::RuntimeStats current = service_->runtime_stats();
        const Clock::time_point now        = Clock::now();
        const ThroughputReport report      = make_throughput_report(
            previous, current, std::chrono::duration<double>(now - previous_time).count());
        if (report_has_activity(report)) { record_throughput(report); }
        previous      = current;
        previous_time = now;
        next_deadline += interval;
        const Clock::time_point after_write = Clock::now();
        if (next_deadline <= after_write) { next_deadline = after_write + interval; }
    }

    const ninfer::RuntimeStats current = service_->runtime_stats();
    const Clock::time_point now        = Clock::now();
    const ThroughputReport tail        = make_throughput_report(
        previous, current, std::chrono::duration<double>(now - previous_time).count());
    // The exact partial interval remains useful to measurement consumers. Pretty throughput is a
    // fixed-cadence operational record and deliberately has no irregular shutdown tail.
    if (report_has_activity(tail)) { request_jsonl_.write_throughput(tail); }
}

void HttpServer::stop_stats_reporter() {
    if (!stats_thread_.joinable()) { return; }
    {
        std::lock_guard lock(stats_mutex_);
        stats_stopping_ = true;
    }
    stats_cv_.notify_one();
    stats_thread_.join();
}

namespace {

// One upstream MCP exchange in flight for the webui's relay. cpp-httplib writes
// this server's status and headers when the route handler returns, but the
// upstream status is only known once the request is on the wire, so the call runs
// on its own thread and the handler blocks until the response head arrives. The
// body then flows through the queue into a chunked provider, which is what keeps
// the long-lived MCP GET channel streaming instead of buffering it whole.
struct McpProxyRelay {
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::string> chunks;
    bool head_ready = false;
    bool finished   = false;
    bool abandoned  = false; // the browser hung up; stop draining upstream
    int status      = 0;
    httplib::Headers response_headers;
    std::string error;
    std::thread worker;

    ~McpProxyRelay() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            abandoned = true;
        }
        if (worker.joinable()) { worker.join(); }
    }
};

void run_mcp_proxy(std::shared_ptr<McpProxyRelay> relay, McpProxyTarget target, std::string method,
                   httplib::Headers headers, std::string body) {
    httplib::Client client(target.host, target.port);
    client.set_connection_timeout(std::chrono::seconds(5));
    // The MCP GET channel stays open for the life of the session and is silent
    // between events, so a short read timeout would tear it down mid-session.
    client.set_read_timeout(std::chrono::hours(1));
    client.set_write_timeout(std::chrono::seconds(30));
    client.set_keep_alive(false);

    httplib::Request upstream;
    upstream.method           = std::move(method);
    upstream.path             = target.path;
    upstream.headers          = std::move(headers);
    upstream.body             = std::move(body);
    upstream.response_handler = [relay](const httplib::Response& response) {
        {
            std::lock_guard<std::mutex> lock(relay->mutex);
            relay->status           = response.status;
            relay->response_headers = response.headers;
            relay->head_ready       = true;
        }
        relay->cv.notify_all();
        return true;
    };
    upstream.content_receiver = [relay](const char* data, std::size_t length, std::uint64_t,
                                        std::uint64_t) {
        {
            std::lock_guard<std::mutex> lock(relay->mutex);
            if (relay->abandoned) { return false; }
            relay->chunks.emplace_back(data, length);
        }
        relay->cv.notify_all();
        return true;
    };

    const httplib::Result result = client.send(upstream);
    {
        std::lock_guard<std::mutex> lock(relay->mutex);
        // A transport failure after the head arrived is a truncated stream, not a
        // failed request: the status is already on its way to the browser.
        if (!result && !relay->head_ready) { relay->error = httplib::to_string(result.error()); }
        relay->finished = true;
    }
    relay->cv.notify_all();
}

// llama.cpp webui dialect: /props is the client's server introspection endpoint
// (role detection, context size, default params, thinking-capability probe, api
// key validation). NInfer has no llama.cpp server behind it, so serve a faithful
// stub derived from the process configuration. Only process-level overrides are
// reported as parameter values; everything else stays at neutral zeros so client
// side defaults never swallow a user-set request parameter.
nlohmann::json make_props_stub(const ServeOptions& options) {
    const auto& ov = options.sampling_overrides;
    nlohmann::json params = nlohmann::json::object();
    params["n_predict"]         = options.default_max_tokens;
    params["seed"]              = ov.seed ? static_cast<double>(*ov.seed) : 0;
    params["temperature"]       = ov.temperature ? static_cast<double>(*ov.temperature) : 0;
    params["dynatemp_range"]    = 0;
    params["dynatemp_exponent"] = 0;
    params["top_k"]             = ov.top_k ? *ov.top_k : 0;
    params["top_p"]             = ov.top_p ? static_cast<double>(*ov.top_p) : 0;
    params["min_p"]             = ov.min_p ? static_cast<double>(*ov.min_p) : 0;
    params["top_n_sigma"]       = 0;
    params["xtc_probability"]   = 0;
    params["xtc_threshold"]     = 0;
    params["typ_p"]             = 0;
    params["repeat_last_n"]     = 0;
    params["repeat_penalty"]    = 0;
    params["presence_penalty"]  = ov.presence_penalty ? static_cast<double>(*ov.presence_penalty) : 0;
    params["frequency_penalty"] =
        ov.frequency_penalty ? static_cast<double>(*ov.frequency_penalty) : 0;
    params["dry_multiplier"]        = 0;
    params["dry_base"]              = 0;
    params["dry_allowed_length"]    = 0;
    params["dry_penalty_last_n"]    = 0;
    params["dry_sequence_breakers"] = nlohmann::json::array();
    params["mirostat"]              = 0;
    params["mirostat_tau"]          = 0;
    params["mirostat_eta"]          = 0;
    params["stop"]                  = nlohmann::json::array();
    params["max_tokens"]            = options.default_max_tokens;
    params["n_keep"]                = 0;
    params["n_discard"]             = 0;
    params["ignore_eos"]            = false;
    params["stream"]                = false;
    params["logit_bias"]            = nlohmann::json::array();
    params["n_probs"]               = 0;
    params["min_keep"]              = 0;
    params["grammar"]               = "";
    params["grammar_lazy"]          = false;
    params["grammar_triggers"]      = nlohmann::json::array();
    params["preserved_tokens"]      = nlohmann::json::array();
    params["chat_format"]           = "";
    params["reasoning_format"]      = "";
    params["reasoning_in_content"]  = false;
    params["generation_prompt"]     = "";
    params["samplers"]              = nlohmann::json::array();
    params["backend_sampling"]      = false;
    params["speculative.n_max"]     = 0;
    params["speculative.n_min"]     = 0;
    params["speculative.p_min"]     = 0.0;
    params["timings_per_token"]     = false;
    params["post_sampling_probs"]   = false;
    params["lora"]                  = nlohmann::json::array();

    nlohmann::json props                 = nlohmann::json::object();
    props["default_generation_settings"] = {
        {"id", 0},
        {"id_task", 0},
        {"n_ctx", static_cast<int>(options.max_context)},
        {"speculative", options.speculative.backend != SpeculativeBackend::None},
        {"is_processing", false},
        {"params", params},
        {"prompt", ""},
        {"next_token",
         {{"has_next_token", false},
          {"has_new_line", false},
          {"n_remain", 0},
          {"n_decoded", 0},
          {"stopping_word", ""}}},
    };
    props["total_slots"] = 1;
    props["model_path"]  = options.artifact_path;
    props["role"]        = "model";
    // The webui ungreys its "Use llama-server proxy" option from this flag alone;
    // without it the relay route is unreachable from the UI even when it is running.
    props["cors_proxy_enabled"] = options.webui_mcp_proxy;
    props["modalities"] = {{"vision", options.enable_vision}, {"audio", false}, {"video", false}};
    // Capability marker only: clients that probe the chat template (e.g. the
    // webui's thinking-support heuristic) need `enable_thinking` to appear; the
    // real template is embedded in the loaded artifact.
    props["chat_template"] =
        "{# ninfer-serve: capability marker; the real chat template is embedded in "
        "the loaded artifact #}\n"
        "{%- if enable_thinking is defined %}\n"
        "  {%- set thinking = enable_thinking %}\n"
        "{% endif %}";
    props["bos_token"]  = "";
    props["eos_token"]  = "";
    props["build_info"] = "ninfer-serve";
    return props;
}

} // namespace

void HttpServer::handle_props(const httplib::Request&, httplib::Response& res) const {
    res.set_content(make_props_stub(options_).dump(), "application/json");
}

void HttpServer::handle_mcp_proxy(const httplib::Request& req, httplib::Response& res) {
    const McpProxyRequest parsed = parse_mcp_proxy_request(req);
    if (!parsed.ok) {
        ApiError error;
        error.status  = 400;
        error.type    = "invalid_request_error";
        error.code    = "invalid_proxy_target";
        error.message = "mcp proxy: " + parsed.error;
        write_openai_error(res, error);
        return;
    }

    auto relay = std::make_shared<McpProxyRelay>();
    relay->worker =
        std::thread(run_mcp_proxy, relay, parsed.target, req.method, parsed.headers, req.body);

    std::unique_lock<std::mutex> lock(relay->mutex);
    relay->cv.wait(lock, [&relay] { return relay->head_ready || relay->finished; });
    if (!relay->head_ready) {
        const std::string reason = relay->error.empty() ? "upstream request failed" : relay->error;
        lock.unlock();
        ApiError error;
        error.status  = 502;
        error.type    = "upstream_error";
        error.code    = "proxy_target_unreachable";
        error.message = "mcp proxy: " + reason;
        write_openai_error(res, error);
        return;
    }
    res.status                              = relay->status;
    const httplib::Headers upstream_headers = relay->response_headers;
    lock.unlock();

    std::string content_type = "application/octet-stream";
    for (const auto& [name, value] : upstream_headers) {
        if (is_hop_by_hop_response_header(name)) { continue; }
        // The chunked provider owns Content-Type; setting it here too would emit
        // the header twice.
        if (header_name_is(name, "content-type")) {
            content_type = value;
            continue;
        }
        res.set_header(name, value);
    }

    res.set_chunked_content_provider(
        content_type, [relay](std::size_t, httplib::DataSink& sink) -> bool {
            std::unique_lock<std::mutex> lock(relay->mutex);
            relay->cv.wait(lock, [&relay] { return !relay->chunks.empty() || relay->finished; });
            std::deque<std::string> ready;
            ready.swap(relay->chunks);
            const bool drained = relay->finished && ready.empty();
            lock.unlock();

            for (const std::string& chunk : ready) {
                if (!sink.write(chunk.data(), chunk.size())) {
                    std::lock_guard<std::mutex> abandon(relay->mutex);
                    relay->abandoned = true;
                    return false;
                }
            }
            if (drained) { sink.done(); }
            return true;
        });
}

void HttpServer::mount_webui(const std::string& webui_dir) {
    std::ifstream index(webui_dir + "/index.html", std::ios::binary);
    if (!index) { throw std::runtime_error("webui dir has no index.html: " + webui_dir); }
    webui_index_html_ =
        std::string((std::istreambuf_iterator<char>(index)), std::istreambuf_iterator<char>());
    webui_serving_ = true;
    if (!server_.set_mount_point("/", webui_dir)) {
        throw std::runtime_error("cannot mount webui directory: " + webui_dir);
    }
    spdlog::info("serving webui from {}", webui_dir);
}

void HttpServer::register_webui_mime() {
    // Wbudowana mapa httplib pokrywa typy zasobow tego UI; przypinamy te, od ktorych
    // zalezy jego ladowanie w przegladarce.
    server_.set_file_extension_and_mimetype_mapping("js", "text/javascript");
    server_.set_file_extension_and_mimetype_mapping("css", "text/css");
    server_.set_file_extension_and_mimetype_mapping("html", "text/html");
    server_.set_file_extension_and_mimetype_mapping("json", "application/json");
    server_.set_file_extension_and_mimetype_mapping("svg", "image/svg+xml");
    server_.set_file_extension_and_mimetype_mapping("ico", "image/x-icon");
}

bool HttpServer::webui_spa_path(const std::string& path) const {
    // Fallback SPA lapie WYLACZNIE trasy klienckie. Zasoby obsluzyl mount statyczny, a
    // sciezki API (/v1/...), /props, /health i pakiety (_app/...) nigdy trasami SPA nie sa:
    // brakujacy plik _app albo nieznana sciezka /v1 musza dalej konczyc sie 404.
    if (path.size() < 2 || path[0] != '/') { return false; }
    if (path == "/") { return false; }
    if (path.rfind("/v1", 0) == 0 && (path.size() == 3 || path[3] == '/')) { return false; }
    if (path == "/props" || path == "/health" || path == "/slots" || path == "/tools" ||
        path == "/v1/streams/lookup" || path == kMcpProxyPath) {
        return false;
    }
    if (path.rfind("/_app/", 0) == 0) { return false; }
    if (path.find_first_of('.') != std::string::npos) { return false; }
    return true;
}

void HttpServer::register_routes() {
    server_.set_error_handler([this](const httplib::Request& request, httplib::Response& response) {
        // Trasy klienckie UI musza dostac powloke aplikacji; prawdziwe zasoby obsluzyl juz
        // mount statyczny, a sciezki API maja zostac przy swoim 404.
        if (webui_serving_ && response.status == 404 && request.method == "GET" &&
            webui_spa_path(request.path)) {
            response.set_content(webui_index_html_, "text/html");
            response.status = 200;
            return httplib::Server::HandlerResponse::Handled;
        }
        return handle_unrendered_http_error(options_, request, response);
    });
    if (options_.enable_cors) {
        server_.set_default_headers(
            {{"Access-Control-Allow-Origin", "*"},
             {"Access-Control-Expose-Headers", "x-request-id, request-id"},
             {"Access-Control-Allow-Headers",
              "Authorization, Content-Type, X-API-Key, anthropic-version, anthropic-beta, "
              "anthropic-user-profile-id"},
             {"Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"}});
        // CORS preflight: browsers send OPTIONS with no credentials before the real
        // request; answer it without auth so the actual GET/POST can carry the key.
        server_.Options(R"(.*)",
                        [](const httplib::Request&, httplib::Response& res) { res.status = 204; });
    }

    server_.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        ensure_openai_request_id(req, res);
        if (options_.api_key.empty() || req.path == "/health" || req.method == "OPTIONS") {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        // Accept both the OpenAI-style bearer token and the Anthropic-style
        // x-api-key header so OpenAI clients and Claude Code (ANTHROPIC_API_KEY
        // -> x-api-key, ANTHROPIC_AUTH_TOKEN -> Authorization: Bearer) both work.
        const bool bearer_ok =
            matches_bearer_credential(req.get_header_value("Authorization"), options_.api_key);
        const bool x_api_key_ok = req.get_header_value("x-api-key") == options_.api_key;
        if (!bearer_ok && !x_api_key_ok) {
            ApiError error;
            error.status  = 401;
            error.type    = "invalid_request_error";
            error.code    = "invalid_api_key";
            error.message = "missing or invalid API key";
            // Render the 401 in the shape the target endpoint speaks.
            if (req.path.rfind("/v1/messages", 0) == 0) {
                write_anthropic_error(res, error, new_anthropic_request_id());
            } else {
                write_openai_error(res, error);
            }
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    server_.set_exception_handler(
        [this](const httplib::Request& req, httplib::Response& res, std::exception_ptr ep) {
            ensure_openai_request_id(req, res);
            try {
                std::rethrow_exception(ep);
            } catch (const ApiException& e) {
                if (e.error().status >= 500) {
                    operational_log_.http_failure(
                        endpoint_name(req.path),
                        make_request_failure(RequestFailurePhase::Http, e.error()),
                        response_request_id(res));
                }
                if (req.path.rfind("/v1/messages", 0) == 0) {
                    write_anthropic_error(res, e.error(), new_anthropic_request_id());
                } else {
                    write_openai_error(res, e.error());
                }
            } catch (const std::exception& e) {
                operational_log_.http_failure(
                    endpoint_name(req.path),
                    make_internal_request_failure(RequestFailurePhase::Http, e.what()),
                    response_request_id(res));
                if (req.path.rfind("/v1/messages", 0) == 0) {
                    ApiError error;
                    error.status  = 500;
                    error.message = e.what();
                    write_anthropic_error(res, error, new_anthropic_request_id());
                } else {
                    write_exception(res, e);
                }
            } catch (...) {
                operational_log_.http_failure(
                    endpoint_name(req.path),
                    make_internal_request_failure(RequestFailurePhase::Http, "unknown error"),
                    response_request_id(res));
                ApiError error;
                error.status  = 500;
                error.type    = "internal_error";
                error.message = "unknown error";
                if (req.path.rfind("/v1/messages", 0) == 0) {
                    write_anthropic_error(res, error, new_anthropic_request_id());
                } else {
                    write_openai_error(res, error);
                }
            }
        });

    server_.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
        const bool available = service_ != nullptr && service_->is_available();
        res.status           = available ? 200 : 503;
        res.set_content(nlohmann::json{{"status", available ? "ok" : "unavailable"}}.dump(),
                        "application/json");
    });
    // llama.cpp webui dialect. /props answers whether or not the UI is served here, so a
    // webui hosted elsewhere can still introspect this server.
    server_.Get("/props", [this](const httplib::Request& req, httplib::Response& res) {
        handle_props(req, res);
    });
    if (options_.webui_mcp_proxy) {
        const auto relay = [this](const httplib::Request& req, httplib::Response& res) {
            handle_mcp_proxy(req, res);
        };
        server_.Get(kMcpProxyPath, relay);
        server_.Post(kMcpProxyPath, relay);
        server_.Delete(kMcpProxyPath, relay);
        spdlog::info("webui MCP relay enabled at {}", kMcpProxyPath);
    }
    server_.Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
        handle_models(req, res);
    });
    server_.Get(R"(/v1/models/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
        handle_model(req, res);
    });
    server_.Post("/v1/chat/completions",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_chat_completions(req, res);
                 });
    server_.Post("/v1/responses", [this](const httplib::Request& req, httplib::Response& res) {
        handle_responses(req, res);
    });
    server_.Post("/v1/responses/input_tokens",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_input_tokens(req, res);
                 });
    server_.Post("/v1/responses/compact",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_compact(req, res);
                 });
    server_.Post(R"(/v1/responses/([^/]+)/cancel)",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_cancel(req, res);
                 });
    server_.Get(R"(/v1/responses/([^/]+)/input_items)",
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_response_input_items(req, res);
                });
    server_.Get(R"(/v1/responses/([^/]+))",
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_response_get(req, res);
                });
    server_.Delete(R"(/v1/responses/([^/]+))",
                   [this](const httplib::Request& req, httplib::Response& res) {
                       handle_response_delete(req, res);
                   });
    server_.Post("/v1/messages/count_tokens",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_count_tokens(req, res);
                 });
    server_.Post("/v1/messages", [this](const httplib::Request& req, httplib::Response& res) {
        handle_messages(req, res);
    });
}

void HttpServer::handle_models(const httplib::Request&, httplib::Response& res) const {
    res.set_content(make_models_list(public_model_id_, unix_time_now(), options_.max_context),
                    "application/json");
}

void HttpServer::handle_model(const httplib::Request& req, httplib::Response& res) const {
    const std::string id = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    if (id != public_model_id_) {
        ApiError error;
        error.status  = 404;
        error.type    = "invalid_request_error";
        error.code    = "model_not_found";
        error.message = "model '" + id + "' not found";
        write_openai_error(res, error);
        return;
    }
    res.set_content(make_model_object(public_model_id_, unix_time_now(), options_.max_context),
                    "application/json");
}

bool HttpServer::bind() { return server_.bind_to_port(options_.host, options_.port); }

void HttpServer::attach(GenerationService& service) {
    if (service_ != nullptr) {
        throw std::logic_error("HTTP generation service is already attached");
    }
    const ninfer::LoadSummary load = service.load_summary();
    public_model_id_               = resolve_public_model_id(options_, load.model_name);
    service_                       = &service;
    request_jsonl_.write_server_start(options_, service.engine_options(),
                                      service.sampling_defaults(), public_model_id_, load,
                                      service.memory_summary());
}

bool HttpServer::listen() {
    if (service_ == nullptr) { throw std::logic_error("HTTP generation service is not attached"); }
    if (public_model_id_.empty()) {
        throw std::logic_error("HTTP public model id is not resolved");
    }
    if (options_.log_stats_interval_ms != 0) {
        stats_stopping_ = false;
        stats_thread_   = std::thread([this] { run_stats_reporter(); });
    }
    try {
        const bool result = server_.listen_after_bind();
        stop_stats_reporter();
        return result;
    } catch (...) {
        stop_stats_reporter();
        throw;
    }
}

void HttpServer::stop() { server_.stop(); }

} // namespace ninfer::serve
