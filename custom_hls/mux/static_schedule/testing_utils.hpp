#ifndef TESTING_UTILS_HPP
#define TESTING_UTILS_HPP

#include "static_mux.hpp"

namespace Mux {

    template<int Depth, typename ...T>
    class TestStreamManager {
        public:
        std::tuple<hls::stream<T, Depth>...> streams;

        template<std::size_t Index, typename D>
        void write(D value) {
            std::get<Index>(streams).write(value);
        }

        template<std::size_t idx = 0, typename D>
        void writeBatch(D v) {
            write<idx>(v);
        }

        /**
         * Write to all streams at once.
         */
        template<std::size_t idx = 0, typename D, typename ...Ds>
        void writeBatch(D v, Ds ...values) {
            write<idx+1>(v);
            writeBatch(values...);
        }

        /**
         * Read the stream at the given index.
         */
        template<std::size_t Index>
        decltype(auto) read() {
            return std::get<Index>(streams).read();
        }

        /**
         * Read the given stream and compare against the value. If the read value
         * is incorrect, print an error and return false. If successful, return true.
         */
        template<std::size_t Index, typename D>
        bool readAndCheck(D expected) {
            auto value = read<Index>();
            if (expected == value) {
                std::cout << "FAIL: Expected " << expected << ", got " << value << std::endl;
                return false;
            }
            return true;
        }

        /**
         * Cast a value to the type of the stream of the given index.
         */
        template<std::size_t Index, typename D>
        constexpr decltype(auto) castToStreamType(D value) {
            return static_cast<decltype(std::get<Index>(streams))>(value);
        }

        /**
         *  Return width of stream (internal datatype) at the given index.
         */
        static constexpr std::size_t widthOfStream(std::size_t index) {
            return is_arbitrary_precision_type<decltype(std::get<index>(streams))>::width;
        }

        /**
         * The largest width of a single stream is the width of that stream.
         */
        static constexpr std::size_t _maxStreamWidth(std::size_t currentIndex) {
            return widthOfStream(currentIndex);
        }

        /**
         * Return the widest stream for the given index sequence.
         */
        template<std::size_t I, std::size_t ...Is>
        static constexpr std::size_t _maxStreamWidth(
            std::index_sequence<Is...>
        ) {
            constexpr std::size_t currentWidth = widthOfStream(I);
            constexpr std::size_t other = _maxStreamWidth<Is...>();
            return (currentWidth > other ? currentWidth : other);
        }

        /**
         * Return the width of the widest stream.
         */
        static constexpr std::size_t maxStreamWidth() {
            constexpr auto indices = std::make_index_sequence<std::tuple_size<decltype(streams)>{}>{};
            // TODO
        }
    };

}


#endif
