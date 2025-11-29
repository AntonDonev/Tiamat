#include "pch.h"
#include <winsock2.h>
#include <windows.h>
#include <ws2tcpip.h>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <winreg.h>

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "Advapi32.lib")

static std::string      g_lastMessage;
static std::mutex       g_messageMutex;
static std::atomic<bool> g_running{ true };
static SOCKET           g_socket = INVALID_SOCKET;
static std::thread      g_recvThread;
static std::string      g_hwid = "";

static const std::string G_ENCRYPTION_KEY = "MMgAZWQi788D8238TjqgPgMhx7XYX4CC";

void recvLoop();
std::string GetMachineGuid();
void LogDebug(const std::string& message);

void LogDebug(const std::string& message) {
    OutputDebugStringA(("[DLL] " + message + "\n").c_str());
}

void XorCipherInPlace(char* data, int len, const std::string& key) {
    if (key.empty() || len <= 0 || data == nullptr) {
        return;
    }
    size_t keyLen = key.length();
    for (int i = 0; i < len; ++i) {
        data[i] = data[i] ^ key[i % keyLen];
    }
}

BOOL APIENTRY DllMain(HMODULE hModule,
    DWORD  ul_reason_for_call,
    LPVOID lpReserved)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    {
        DisableThreadLibraryCalls(hModule);

        if (G_ENCRYPTION_KEY.empty()) {
            LogDebug("WARNING: Encryption key is not set or is empty! Communication will NOT be effectively encrypted.");
        }
        else {
            LogDebug("Encryption key initialized. Length: " + std::to_string(G_ENCRYPTION_KEY.length()));
        }

        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
            LogDebug("WSAStartup failed. Error: " + std::to_string(WSAGetLastError()));
            return FALSE;
        }

        g_hwid = GetMachineGuid();
        if (g_hwid.empty()) {
            LogDebug("Failed to get Machine GUID (HWID). Authentication will likely fail.");
            std::lock_guard<std::mutex> lock(g_messageMutex);
            g_lastMessage = "HWID_DLL|FAILED_TO_RETRIEVE";
            LogDebug("Initial HWID message for EA prepared: " + g_lastMessage);
        }
        else {
            LogDebug("Machine GUID (HWID) retrieved: " + g_hwid);
            std::lock_guard<std::mutex> lock(g_messageMutex);
            g_lastMessage = "HWID_DLL|" + g_hwid;
            LogDebug("Initial HWID message for EA prepared: " + g_lastMessage);
        }

        g_running = true;
        g_recvThread = std::thread(recvLoop);
        LogDebug("DLL Attached and receive thread started.");
        break;
    }
    case DLL_PROCESS_DETACH:
    {
        LogDebug("DLL Detaching...");
        g_running = false;

        if (g_socket != INVALID_SOCKET) {
            shutdown(g_socket, SD_BOTH);
            closesocket(g_socket);
            g_socket = INVALID_SOCKET;
            LogDebug("Socket closed.");
        }

        if (g_recvThread.joinable()) {
            g_recvThread.join();
            LogDebug("Receive thread joined.");
        }

        WSACleanup();
        LogDebug("DLL Detached.");
        break;
    }
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
        break;
    }
    return TRUE;
}

std::string GetMachineGuid()
{
    std::string guid_str = "";
    HKEY hKey = NULL;
    LONG lResult = RegOpenKeyExA(HKEY_LOCAL_MACHINE, "SOFTWARE\\Microsoft\\Cryptography", 0, KEY_READ | KEY_WOW64_64KEY, &hKey);

    if (lResult != ERROR_SUCCESS) {
        lResult = RegOpenKeyExA(HKEY_LOCAL_MACHINE, "SOFTWARE\\Microsoft\\Cryptography", 0, KEY_READ, &hKey);
    }

    if (lResult == ERROR_SUCCESS)
    {
        CHAR value[256];
        DWORD size = sizeof(value);
        lResult = RegQueryValueExA(hKey, "MachineGuid", NULL, NULL, (LPBYTE)value, &size);

        if (lResult == ERROR_SUCCESS)
        {
            guid_str = value;
        }
        else
        {
            LogDebug("Failed to query MachineGuid value. Error code: " + std::to_string(lResult));
        }
        RegCloseKey(hKey);
    }
    else {
        LogDebug("Failed to open Cryptography registry key. Error code: " + std::to_string(lResult));
    }
    return guid_str;
}

void recvLoop()
{
    const wchar_t* kServerIP = L"34.134.251.3";
    const unsigned int kServerPort = 12345;

    char buffer[1024] = { 0 };
    DWORD lastConnectAttemptTick = 0;
    const DWORD connectRetryDelay = 30000;
    bool needsAuth = false;

    LogDebug("Receive loop started.");
    if (G_ENCRYPTION_KEY.empty()) {
        LogDebug("Receive loop: WARNING - Encryption key is empty or using placeholder. Communication will be unencrypted or insecure.");
    }
    else {
        LogDebug("Receive loop: Encryption key is set. Encrypted communication expected from server.");
    }

    while (g_running)
    {
        if (g_socket == INVALID_SOCKET)
        {
            DWORD now = GetTickCount();
            if (now - lastConnectAttemptTick >= connectRetryDelay || lastConnectAttemptTick == 0)
            {
                if (lastConnectAttemptTick > 0 && now < lastConnectAttemptTick) {
                    lastConnectAttemptTick = 0;
                }

                std::wstring serverDetailsW = std::wstring(kServerIP) + L":" + std::to_wstring(kServerPort);
                std::string serverDetailsA;
                if (!serverDetailsW.empty()) {
                    int bufferSize = WideCharToMultiByte(CP_UTF8, 0, serverDetailsW.c_str(), -1, nullptr, 0, nullptr, nullptr);
                    if (bufferSize > 0) {
                        serverDetailsA.resize(bufferSize - 1);
                        WideCharToMultiByte(CP_UTF8, 0, serverDetailsW.c_str(), -1, &serverDetailsA[0], bufferSize, nullptr, nullptr);
                    }
                    else {
                        serverDetailsA = "[conversion_error_ip_port]";
                        LogDebug("WideCharToMultiByte failed for server IP/Port. Error: " + std::to_string(GetLastError()));
                    }
                }
                else {
                    serverDetailsA = "[empty_server_details]";
                }
                LogDebug("Attempting to connect to server " + serverDetailsA);
                lastConnectAttemptTick = now;

                SOCKET tempSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
                if (tempSocket == INVALID_SOCKET)
                {
                    LogDebug("socket() failed. Error: " + std::to_string(WSAGetLastError()));
                    Sleep(1000);
                    continue;
                }

                sockaddr_in server_addr;
                server_addr.sin_family = AF_INET;
                server_addr.sin_port = htons(kServerPort);
                int ptonResult = InetPton(AF_INET, kServerIP, &server_addr.sin_addr);

                if (ptonResult != 1) {
                    LogDebug("InetPton failed for IP " + std::string(serverDetailsA.begin(), serverDetailsA.end()) + ". Error: " + std::to_string(WSAGetLastError()));
                    closesocket(tempSocket);
                    Sleep(connectRetryDelay);
                    continue;
                }

                if (connect(tempSocket, (struct sockaddr*)&server_addr, sizeof(server_addr)) == 0)
                {
                    g_socket = tempSocket;
                    needsAuth = true;
                    LogDebug("Connected to server successfully.");
                }
                else
                {
                    LogDebug("connect() failed. Error: " + std::to_string(WSAGetLastError()) + ". Retrying later.");
                    closesocket(tempSocket);
                }
            }
            if (g_socket == INVALID_SOCKET) {
                Sleep(1000);
                continue;
            }
        }

        if (needsAuth && g_socket != INVALID_SOCKET) {
            needsAuth = false;
            if (!g_hwid.empty()) {
                std::string authMsg = "AUTH|HWID=" + g_hwid;
                int sendResult = send(g_socket, authMsg.c_str(), static_cast<int>(authMsg.length()), 0);
                if (sendResult == SOCKET_ERROR) {
                    LogDebug("send() AUTH message failed. Error: " + std::to_string(WSAGetLastError()));
                    closesocket(g_socket);
                    g_socket = INVALID_SOCKET;
                    continue;
                }
                else {
                    LogDebug("Sent AUTH message: " + authMsg);
                }
            }
            else {
                LogDebug("HWID is empty, cannot send AUTH message. Connection will likely be rejected by server.");
            }
        }

        if (g_socket != INVALID_SOCKET) {
            fd_set readfds;
            FD_ZERO(&readfds);
            FD_SET(g_socket, &readfds);

            timeval tv;
            tv.tv_sec = 0;
            tv.tv_usec = 200000;

            int selectResult = select(0, &readfds, NULL, NULL, &tv);

            if (!g_running) break;

            if (selectResult > 0 && FD_ISSET(g_socket, &readfds))
            {
                memset(buffer, 0, sizeof(buffer));
                int recvLen = recv(g_socket, buffer, sizeof(buffer) - 1, 0);
                if (recvLen > 0)
                {
                    if (!G_ENCRYPTION_KEY.empty()) {
                        XorCipherInPlace(buffer, recvLen, G_ENCRYPTION_KEY);
                    }
                    else if (!G_ENCRYPTION_KEY.empty()) {
                        XorCipherInPlace(buffer, recvLen, G_ENCRYPTION_KEY);
                        LogDebug("Warning: Data was XORed with a placeholder key.");
                    }

                    buffer[recvLen] = '\0';
                    std::string receivedMsg(buffer);
                    LogDebug("Received (potentially decrypted) message: " + receivedMsg);
                    {
                        std::lock_guard<std::mutex> lock(g_messageMutex);
                        g_lastMessage = receivedMsg;
                    }
                }
                else if (recvLen == 0)
                {
                    LogDebug("Server closed connection.");
                    closesocket(g_socket);
                    g_socket = INVALID_SOCKET;
                }
                else
                {
                    int error = WSAGetLastError();
                    LogDebug("recv() failed. Error: " + std::to_string(error) + ". Closing socket.");
                    closesocket(g_socket);
                    g_socket = INVALID_SOCKET;
                }
            }
            else if (selectResult < 0) {
                int error = WSAGetLastError();
                LogDebug("select() failed. Error: " + std::to_string(error) + ". Closing socket.");
                closesocket(g_socket);
                g_socket = INVALID_SOCKET;
            }
        }
        Sleep(50);
    }
    LogDebug("Receive loop finished.");
}

extern "C" __declspec(dllexport)
const char* __cdecl DllGetMessage()
{
    static thread_local std::string localCopy;
    {
        std::lock_guard<std::mutex> lock(g_messageMutex);
        if (!g_lastMessage.empty()) {
            localCopy = g_lastMessage;
            g_lastMessage.clear();
        }
        else {
            localCopy = "";
        }
    }
    return localCopy.c_str();
}

extern "C" __declspec(dllexport)
const wchar_t* __cdecl DllGetMessageW()
{
    static thread_local std::wstring localCopyW;
    std::string temp;
    {
        std::lock_guard<std::mutex> lock(g_messageMutex);
        if (!g_lastMessage.empty()) {
            temp = g_lastMessage;
            g_lastMessage.clear();
        }
    }

    if (!temp.empty())
    {
        int needed = MultiByteToWideChar(CP_UTF8, 0, temp.c_str(), -1, nullptr, 0);
        if (needed > 0)
        {
            localCopyW.resize(needed - 1);
            MultiByteToWideChar(CP_UTF8, 0, temp.c_str(), -1, &localCopyW[0], needed);
        }
        else
        {
            LogDebug("MultiByteToWideChar failed to calculate size or convert in DllGetMessageW. Error: " + std::to_string(GetLastError()));
            localCopyW = L"";
        }
    }
    else
    {
        localCopyW = L"";
    }
    return localCopyW.c_str();
}

extern "C" __declspec(dllexport)
void __cdecl SendMessageToServer(const char* msg)
{
    if (!msg || strlen(msg) == 0) {
        LogDebug("SendMessageToServer: Attempted to send null or empty message.");
        return;
    }
    if (g_socket == INVALID_SOCKET) {
        LogDebug("SendMessageToServer: Socket invalid, cannot send message: " + std::string(msg));
        return;
    }

    std::string toSend(msg);

    LogDebug("Attempting to send message: " + toSend);
    int sendResult = send(g_socket, toSend.c_str(), static_cast<int>(toSend.length()), 0);

    if (sendResult == SOCKET_ERROR) {
        int error = WSAGetLastError();
        LogDebug("send() failed in SendMessageToServer. Error: " + std::to_string(error) + ". Closing socket.");
        closesocket(g_socket);
        g_socket = INVALID_SOCKET;
    }
    else {
        LogDebug("Successfully sent: " + toSend);
    }
}

extern "C" __declspec(dllexport)
void __cdecl SendMessageToServerW(const wchar_t* msg)
{
    if (!msg || wcslen(msg) == 0) {
        LogDebug("SendMessageToServerW: Attempted to send null or empty wide message.");
        return;
    }

    int needed = WideCharToMultiByte(CP_UTF8, 0, msg, -1, nullptr, 0, nullptr, nullptr);
    if (needed <= 1) {
        LogDebug("WideCharToMultiByte failed to calculate size or empty wide string in SendMessageToServerW. Error: " + std::to_string(GetLastError()));
        if (needed == 1 && wcslen(msg) == 0) {
            LogDebug("SendMessageToServerW: Attempted to send an empty (but not null) wide message.");
        }
        return;
    }

    std::string narrowMsg;
    narrowMsg.resize(needed - 1);
    if (WideCharToMultiByte(CP_UTF8, 0, msg, -1, &narrowMsg[0], needed, nullptr, nullptr) == 0) {
        LogDebug("WideCharToMultiByte failed during conversion in SendMessageToServerW. Error: " + std::to_string(GetLastError()));
        return;
    }
    SendMessageToServer(narrowMsg.c_str());
}

extern "C" __declspec(dllexport)
void __cdecl RunDllEntry(HWND hwnd, HINSTANCE hinst, LPSTR lpszCmdLine, int nCmdShow)
{
    LogDebug("RunDllEntry called. Command line: " + (lpszCmdLine ? std::string(lpszCmdLine) : "null"));
    std::string hwid_test = GetMachineGuid();
    if (!hwid_test.empty()) {
        LogDebug("RunDllEntry - HWID Test: " + hwid_test);
    }
    else {
        LogDebug("RunDllEntry - HWID Test: Failed to get HWID.");
    }

    if (!G_ENCRYPTION_KEY.empty() && G_ENCRYPTION_KEY != "YOUR_SECRET_KEY_HERE") {
        LogDebug("RunDllEntry - Encryption Key Test: Initialized, length " + std::to_string(G_ENCRYPTION_KEY.length()));
    }
    else {
        LogDebug("RunDllEntry - Encryption Key Test: NOT SET or using placeholder.");
    }

    MessageBoxA(nullptr, "DLL Loaded via rundll32! Check debug output for logs.", "SignalProvider DLL Test", MB_OK | MB_ICONINFORMATION);
}