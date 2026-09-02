#include "serve/mcp_proxy.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string_view>

namespace ninfer::serve {
namespace {

std::string to_lower(std::string_view text) {
    std::string lowered(text);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lowered;
}

bool starts_with_ci(const std::string& text, std::string_view prefix) {
    if (text.size() < prefix.size()) { return false; }
    return to_lower(std::string_view(text).substr(0, prefix.size())) == to_lower(prefix);
}

McpProxyRequest reject(std::string message) {
    McpProxyRequest request;
    request.ok    = false;
    request.error = std::move(message);
    return request;
}

} // namespace

McpProxyRequest parse_mcp_proxy_target(const std::string& url) {
    if (url.empty()) { return reject("proxy target url is empty"); }
    if (starts_with_ci(url, "https://")) {
        return reject("https proxy targets are unsupported: this build has no TLS client");
    }
    if (!starts_with_ci(url, "http://")) {
        return reject("proxy target url must be an absolute http:// url");
    }

    const std::string_view rest = std::string_view(url).substr(std::string_view("http://").size());
    const std::size_t authority_end = rest.find_first_of("/?#");
    const std::string_view authority =
        authority_end == std::string_view::npos ? rest : rest.substr(0, authority_end);
    if (authority.empty()) { return reject("proxy target url has no host"); }

    McpProxyRequest request;
    request.ok = true;

    // A bracketed IPv6 literal keeps its colons; only a colon after the closing
    // bracket (or in a plain host) introduces the port.
    std::string_view host_part = authority;
    std::string_view port_part;
    const std::size_t port_colon = authority.front() == '['
                                       ? authority.find(':', authority.find(']') + 1)
                                       : authority.rfind(':');
    if (port_colon != std::string_view::npos && authority.front() != '[') {
        // Reject a bare IPv6 literal written without brackets rather than reading
        // its last group as a port.
        if (authority.find(':') != port_colon) { return reject("proxy target host is malformed"); }
    }
    if (port_colon != std::string_view::npos) {
        host_part = authority.substr(0, port_colon);
        port_part = authority.substr(port_colon + 1);
    }
    if (host_part.size() >= 2 && host_part.front() == '[' && host_part.back() == ']') {
        host_part = host_part.substr(1, host_part.size() - 2);
    }
    if (host_part.empty()) { return reject("proxy target url has no host"); }

    if (!port_part.empty()) {
        if (!std::all_of(port_part.begin(), port_part.end(),
                         [](unsigned char c) { return std::isdigit(c) != 0; })) {
            return reject("proxy target port is not a number");
        }
        const long port = std::strtol(std::string(port_part).c_str(), nullptr, 10);
        if (port <= 0 || port > 65535) { return reject("proxy target port is out of range"); }
        request.target.port = static_cast<int>(port);
    }

    request.target.host = std::string(host_part);
    // The fragment is client-side only and is never sent upstream.
    std::string_view path =
        authority_end == std::string_view::npos ? std::string_view() : rest.substr(authority_end);
    const std::size_t fragment = path.find('#');
    if (fragment != std::string_view::npos) { path = path.substr(0, fragment); }
    request.target.path = path.empty() ? "/" : std::string(path);
    return request;
}

McpProxyRequest parse_mcp_proxy_request(const httplib::Request& request) {
    if (!request.has_param(kMcpProxyUrlParam)) {
        return reject(std::string("missing ") + kMcpProxyUrlParam + " query parameter");
    }

    McpProxyRequest parsed = parse_mcp_proxy_target(request.get_param_value(kMcpProxyUrlParam));
    if (!parsed.ok) { return parsed; }

    const std::size_t prefix_length = std::string_view(kMcpProxyHeaderPrefix).size();
    for (const auto& [name, value] : request.headers) {
        if (!starts_with_ci(name, kMcpProxyHeaderPrefix)) { continue; }
        std::string forwarded = name.substr(prefix_length);
        if (forwarded.empty()) { continue; }
        // httplib::Client owns Host and the request framing for the upstream hop.
        const std::string lowered = to_lower(forwarded);
        if (lowered == "host" || lowered == "content-length" || lowered == "connection" ||
            lowered == "transfer-encoding") {
            continue;
        }
        parsed.headers.emplace(std::move(forwarded), value);
    }
    return parsed;
}

bool header_name_is(const std::string& name, std::string_view expected) {
    return to_lower(name) == expected;
}

bool is_hop_by_hop_response_header(const std::string& name) {
    const std::string lowered = to_lower(name);
    if (lowered.rfind("access-control-", 0) == 0) { return true; }
    return lowered == "connection" || lowered == "keep-alive" || lowered == "transfer-encoding" ||
           lowered == "content-length" || lowered == "upgrade" || lowered == "proxy-authenticate" ||
           lowered == "proxy-authorization" || lowered == "te" || lowered == "trailer";
}

} // namespace ninfer::serve
