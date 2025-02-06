// SignalProvider.cpp

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

static std::string g_lastMessage;
static std::mutex  g_messageMutex;
static std::atomic<bool> g_running{ true };
static SOCKET g_socket = INVALID_SOCKET;
static std::thread g_recvThread;

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
        {
            return FALSE; // fail
        }

        sockaddr_in server;
        server.sin_family = AF_INET;
        server.sin_port = htons(12345); // must match Python server's port

        if (InetPton(AF_INET, L"34.163.5.184", &server.sin_addr) != 1)
        {
            WSACleanup();
            return FALSE;
        }

        g_socket = socket(AF_INET, SOCK_STREAM, 0);
        if (g_socket == INVALID_SOCKET)
        {
            WSACleanup();
            return FALSE;
        }

        int res = connect(g_socket, (struct sockaddr*)&server, sizeof(server));
        if (res < 0)
        {
            closesocket(g_socket);
            g_socket = INVALID_SOCKET;
        }
        else
        {
            // Start receive thread
            g_running = true;
            g_recvThread = std::thread(recvLoop);
        }
    }
    break;

    case DLL_PROCESS_DETACH:
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

    return TRUE;
}

// Continuously receive data from the server
void recvLoop()
{
    char buffer[1024];
    while (g_running)
    {
        if (g_socket == INVALID_SOCKET)
        {
            Sleep(100);
            continue;
        }

        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(g_socket, &readfds);

        timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 200000; // 200ms

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
                // Server closed connection
                break;
            }
            else
            {
                // Socket error
                int err = WSAGetLastError();
                break;
            }
        }
        Sleep(50);
    }
}

// Exported function for MQL (or other apps) to retrieve the latest message
extern "C" __declspec(dllexport)
const char* __stdcall DllGetMessage()
{
    static thread_local std::string localCopy;
    {
        std::lock_guard<std::mutex> lock(g_messageMutex);
        localCopy = g_lastMessage;
        // Clear so we don't return the same message
        g_lastMessage.clear();
    }
    return localCopy.c_str();
}

// Exported function for MQL (or other apps) to send a message to Python
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

