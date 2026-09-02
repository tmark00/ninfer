#pragma once

#include <httplib.h>

#include <string>
#include <string_view>

namespace ninfer::serve {

// Same-origin relay for the webui's MCP client. A browser page cannot talk to a
// third-party MCP server directly: the MCP transports answer POST from a raw
// response writer, so the reply carries no Access-Control-Allow-Origin even when
// the preflight succeeds. The webui therefore sends the request to its own origin
// under kMcpProxyPath and the server forwards it, which takes CORS out of the
// picture entirely. These three constants are the webui's wire contract and must
// match it exactly.
inline constexpr const char* kMcpProxyPath         = "/cors-proxy";
inline constexpr const char* kMcpProxyUrlParam     = "url";
inline constexpr const char* kMcpProxyHeaderPrefix = "x-llama-server-proxy-header-";

struct McpProxyTarget {
    std::string host;
    int port = 80;
    std::string path; // path plus query, ready for httplib::Client
};

// Parsed relay request, or the reason it must be rejected.
struct McpProxyRequest {
    bool ok = false;
    std::string error; // non-empty exactly when !ok; rendered as a 400
    McpProxyTarget target;
    httplib::Headers headers; // forwarded upstream, prefix already stripped
};

// Splits an absolute http:// URL into the pieces httplib::Client needs. https is
// rejected rather than silently downgraded: the vendored httplib is built without
// CPPHTTPLIB_OPENSSL_SUPPORT, so this build has no TLS client.
McpProxyRequest parse_mcp_proxy_target(const std::string& url);

// Full request parse: target from the url query parameter, forwarded headers from
// the prefixed ones. Headers without the prefix belong to the hop between browser
// and this server and are dropped.
McpProxyRequest parse_mcp_proxy_request(const httplib::Request& request);

// Case-insensitive header-name match. `expected` must already be lowercase.
bool header_name_is(const std::string& name, std::string_view expected);

// True for headers that must not be copied from the upstream response onto ours.
// Hop-by-hop framing belongs to the upstream connection, and the CORS headers
// would collide with the ones this server emits under --cors: a browser rejects
// "Access-Control-Allow-Origin: *, *" outright.
bool is_hop_by_hop_response_header(const std::string& name);

} // namespace ninfer::serve
