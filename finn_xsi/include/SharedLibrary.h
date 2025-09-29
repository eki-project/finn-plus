#ifndef SHAREDLIBRARY_H_
#define SHAREDLIBRARY_H_

#include <optional>
#include <string>
#include <utility>

#if defined(_WIN32)
    #include <windows.h>
#else
    #include <dlfcn.h>
#endif

namespace xsi {
    class SharedLibrary {
         public:
        static char const library_suffix[];

         private:
        using handle_type =
#if defined(_WIN32)
            HINSTANCE;
#else
            void*;
#endif

        //-----------------------------------------------------------------------
        // Instance State
         private:
        handle_type _lib;
        std::string _path;

        //-----------------------------------------------------------------------
        // Life Cycle
         public:
        SharedLibrary();
        SharedLibrary(std::string const& path);
        ~SharedLibrary();

         private:
        SharedLibrary(SharedLibrary const&) = delete;
        SharedLibrary& operator=(SharedLibrary const&) = delete;

         public:
        // Move constructor
        SharedLibrary(SharedLibrary&& other) noexcept;

        // Move assignment operator
        SharedLibrary& operator=(SharedLibrary&& other) noexcept;

         public:
        operator bool() const;
        SharedLibrary& open(std::string const& path);
        SharedLibrary& close();

         private:
        static handle_type load(std::string const& path);
        void unload();

        //-----------------------------------------------------------------------
        // Accessors
         public:
        std::string const& path() const;
        std::optional<void*> getsymbol(char const* const name);

    };  // class SharedLibrary
}  // namespace xsi

#endif /* SHAREDLIBRARY_H_ */
