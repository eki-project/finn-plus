#include <SharedLibrary.h>

#include <stdexcept>

using namespace xsi;

char const SharedLibrary::library_suffix[] =
#if defined(_WIN32)
    ".lib";
#else
    ".so";
#endif

#if defined(_WIN32)
namespace {
    std::string translate_error_message(DWORD errid) {
        std::string msg;
        LPTSTR bufptr;
        FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, errid, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), &bufptr, 0, nullptr);
        if (bufptr)
            msg = reinterpret_cast<char*>(bufptr);
        LocalFree(bufptr);
        return msg;
    }
}  // namespace
#endif

SharedLibrary& SharedLibrary::open(const std::string& path) {
    if (_lib)
        throw std::runtime_error("SharedLibrary still open for " + _path);
    _lib = load(path);
    _path = path;
    return *this;
}

SharedLibrary::handle_type SharedLibrary::load(const std::string& path) {
    if (path.empty())
        throw std::domain_error("Empty library path.");

#if defined(_WIN32)
    SetLastError(0);
    #ifdef UNICODE
    // Use LoadLibraryA explicitly on windows if UNICODE is defined
    handle_type const lib = LoadLibraryA(path.c_str());
    #else
    handle_type const lib = LoadLibrary(path.c_str());
    #endif
    if (!lib)
        throw std::runtime_error(translate_error_message(GetLastError()));
#else
    handle_type const lib = dlopen(path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    if (!lib)
        throw std::runtime_error(dlerror());
#endif
    return lib;
}

void SharedLibrary::unload() noexcept {
    if (_lib) {
#if defined(_WIN32)
        FreeLibrary(_lib);
#else
        dlclose(_lib);
#endif
    }
}

std::optional<void*> SharedLibrary::getsymbol(const char* const name) {
    void* sym;
#if defined(_WIN32)
    sym = (void*) GetProcAddress(_lib, name);
    if (!sym)
#else
    dlerror();  // clear error
    sym = dlsym(_lib, name);
    char const* const err = dlerror();
    if (err)
#endif
        return std::nullopt;
    return std::make_optional(sym);
}

// Constructors
SharedLibrary::SharedLibrary() : _lib(nullptr), _path() {}

SharedLibrary::SharedLibrary(const std::string& path) : _lib(load(path)), _path(path) {}

// Destructor
SharedLibrary::~SharedLibrary() { unload(); }

// Move constructor
SharedLibrary::SharedLibrary(SharedLibrary&& other) noexcept : _lib(other._lib), _path(std::move(other._path)) { other._lib = nullptr; }

// Move assignment operator
SharedLibrary& SharedLibrary::operator=(SharedLibrary&& other) noexcept {
    if (this != &other) {
        // Clean up current state
        unload();

        // Move from other
        _lib = other._lib;
        _path = std::move(other._path);

        // Reset other
        other._lib = nullptr;
    }
    return *this;
}

// Member functions
SharedLibrary::operator bool() const noexcept { return bool(_lib); }

SharedLibrary& SharedLibrary::close() noexcept {
    unload();
    _lib = nullptr;
    _path.clear();
    return *this;
}

const std::string& SharedLibrary::path() const noexcept { return _path; }
