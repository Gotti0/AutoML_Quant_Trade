/// <reference types="vite/client" />

interface ImportMetaEnv {
    readonly VITE_USE_MOCK_DATA: string
    // more env variables...
}

interface ImportMeta {
    readonly env: ImportMetaEnv
}
