
#include "pch.h"
#include <winsock2.h>
#include <windows.h>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <iostream>
#include <Ws2tcpip.h>

#pragma comment(lib, "ws2_32.lib")

static std::string       g_lastMessage;
static std::mutex        g_messageMutex;
static std::atomic<bool> g_running{ true };
static SOCKET            g_socket = INVALID_SOCKET;
static std::thread       g_recvThread;

void recvLoop();

BOOL APIENTRY DllMain(HMODULE hModule,
    DWORD  ul_reason_for_call,
    LPVOID lpReserved)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    {
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
            return FALSE;

        g_running = true;
        g_recvThread = std::thread(recvLoop);

        break;
    }
    case DLL_PROCESS_DETACH:
    {
        g_running = false;
        if (g_recvThread.joinable())
            g_recvThread.join();

        if (g_socket != INVALID_SOCKET)
        {
            closesocket(g_socket);
            g_socket = INVALID_SOCKET;
        }

        WSACleanup();
        break;
    }
    }
    return TRUE;
}

void recvLoop()
{
    const wchar_t* kServerIP = L"34.174.186.157";
    const unsigned int kServerPort = 12345;

    char buffer[1024] = { 0 };

    DWORD lastConnectAttempt = 0;

    while (g_running)
    {
        if (g_socket == INVALID_SOCKET)
        {
            DWORD now = GetTickCount();
            if (now - lastConnectAttempt >= 60000)
            {
                lastConnectAttempt = now;

                SOCKET tempSocket = socket(AF_INET, SOCK_STREAM, 0);
                if (tempSocket == INVALID_SOCKET)
                {
                    Sleep(1000);
                    continue;
                }

                sockaddr_in server;
                server.sin_family = AF_INET;
                server.sin_port = htons(kServerPort);
                if (InetPton(AF_INET, kServerIP, &server.sin_addr) != 1)
                {
                    closesocket(tempSocket);
                    Sleep(60000);
                    continue;
                }

                if (connect(tempSocket, (struct sockaddr*)&server, sizeof(server)) == 0)
                {
                    g_socket = tempSocket;
                    std::cout << "[recvLoop] Connected to server.\n";
                }
                else
                {
                    closesocket(tempSocket);
                    std::cout << "[recvLoop] Connect failed. Will retry in 1 minute.\n";
                }
            }

            if (g_socket == INVALID_SOCKET)
            {
                Sleep(1000);
                continue;
            }
        }

        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(g_socket, &readfds);

        timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 200000; 

        int sel = select(0, &readfds, NULL, NULL, &tv);
        if (sel > 0 && FD_ISSET(g_socket, &readfds))
        {
            int recvLen = recv(g_socket, buffer, sizeof(buffer) - 1, 0);
            if (recvLen > 0)
            {
                buffer[recvLen] = '\0';
                std::string msg(buffer);

                {
                    std::lock_guard<std::mutex> lock(g_messageMutex);
                    g_lastMessage = msg;
                }
            }
            else if (recvLen == 0)
            {
                std::cout << "[recvLoop] Server closed connection.\n";
                closesocket(g_socket);
                g_socket = INVALID_SOCKET;
            }
            else
            {
                int err = WSAGetLastError();
                std::cout << "[recvLoop] Recv error " << err << ". Closing socket.\n";
                closesocket(g_socket);
                g_socket = INVALID_SOCKET;
            }
        }

        Sleep(50);
    }
}

extern "C" __declspec(dllexport)
const char* __stdcall DllGetMessage()
{
    static thread_local std::string localCopy;

    {
        std::lock_guard<std::mutex> lock(g_messageMutex);
        localCopy = g_lastMessage;
        g_lastMessage.clear();
    }
    return localCopy.c_str();
}

extern "C" __declspec(dllexport)
const wchar_t* __stdcall DllGetMessageW()
{
    static thread_local std::wstring localCopyW;

    std::string temp;
    {
        std::lock_guard<std::mutex> lock(g_messageMutex);
        temp = g_lastMessage;
        g_lastMessage.clear();
    }

    if (!temp.empty())
    {
        int needed = MultiByteToWideChar(CP_UTF8, 0, temp.c_str(), -1, nullptr, 0);
        if (needed > 0)
        {
            localCopyW.resize(needed);
            MultiByteToWideChar(CP_UTF8, 0, temp.c_str(), -1, &localCopyW[0], needed);
        }
        else
        {
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
void __stdcall SendMessageToServer(const char* msg)
{
    if (!msg) return;
    if (g_socket == INVALID_SOCKET) return;

    std::string toSend(msg);
    send(g_socket, toSend.c_str(), static_cast<int>(toSend.size()), 0);
}

extern "C" __declspec(dllexport)
void CALLBACK RunDllEntry(HWND hwnd, HINSTANCE hinst, LPSTR lpszCmdLine, int nCmdShow)
{
    MessageBoxA(
        nullptr,
        "rundll32!",
        "SignalProvider",
        MB_OK
    );
}
