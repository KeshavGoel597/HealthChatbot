'use client'

import { useState, useRef, useEffect } from 'react'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Separator } from "@/components/ui/separator"
import { Send, User, Bot, Loader2, Settings, MoreVertical, Menu, PanelLeftClose, PanelLeftOpen, Languages, Play, Volume2 } from 'lucide-react'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuLabel,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu"
import { ModeToggle } from "@/components/mode-toggle"
import { ChatSidebar } from "@/components/chat/chat-sidebar"

const BACKEND_URL = 'http://localhost:8013'

interface Message {
  role: 'user' | 'assistant'
  content: string
  audio_content?: string
}

interface ChatSession {
  id: string
  title: string
  created_at: string
}

const LANGUAGES = [
  { code: "en-IN", name: "English" },
  { code: "hi-IN", name: "Hindi" },
  { code: "ta-IN", name: "Tamil" },
  { code: "te-IN", name: "Telugu" },
  { code: "ml-IN", name: "Malayalam" },
  { code: "kn-IN", name: "Kannada" },
  { code: "mr-IN", name: "Marathi" },
  { code: "gu-IN", name: "Gujarati" },
  { code: "bn-IN", name: "Bengali" },
  { code: "pa-IN", name: "Punjabi" },
  { code: "or-IN", name: "Odia" },
]

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([
    { role: 'assistant', content: 'Hello! I am Robert. I have access to your medical records. How can I help you understand them today?' }
  ])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [model, setModel] = useState<string | null>("gemini-2.5-flash-lite") // "gemini-2.5-flash-lite" = Gemini (default), null = HF local
  const [language, setLanguage] = useState<string>("en-IN")
  const scrollAreaRef = useRef<HTMLDivElement>(null)

  // Sidebar & Session State
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)

  // Fetch sessions on mount
  useEffect(() => {
    fetchSessions()
  }, [])

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }
    }
  }, [messages])

  const fetchSessions = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/sessions/patient101`)
      if (res.ok) {
        const data = await res.json()
        setSessions(data)
      }
    } catch (error) {
      console.error("Failed to fetch sessions:", error)
    }
  }

  const handleNewChat = () => {
    setCurrentSessionId(null)
    setMessages([
      { role: 'assistant', content: 'Hello! I am Robert. I have access to your medical records. How can I help you understand them today?' }
    ])
    // On mobile, close sidebar after action?
    if (window.innerWidth < 1024) setIsSidebarOpen(false)
  }

  const handleSelectSession = async (sessionId: string) => {
    setIsLoading(true)
    try {
      const res = await fetch(`${BACKEND_URL}/sessions/${sessionId}/messages`)
      if (res.ok) {
        const data = await res.json()
        setMessages(data.messages || [])
        setCurrentSessionId(sessionId)
      }
    } catch (error) {
      console.error("Failed to load session:", error)
    } finally {
      setIsLoading(false)
      if (window.innerWidth < 1024) setIsSidebarOpen(false)
    }
  }

  const handleSend = async () => {
    if (!input.trim()) return

    const userMessage = { role: 'user' as const, content: input }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    try {
      let sessionId = currentSessionId

      // If no session, create one first
      if (!sessionId) {
        const createRes = await fetch(`${BACKEND_URL}/sessions/patient101`, { method: 'POST' })
        if (createRes.ok) {
          const newSession = await createRes.json()
          sessionId = newSession.id
          setCurrentSessionId(sessionId!)
          // Refresh sessions list to show the new one immediately (or optimise to add locally)
          fetchSessions()
        } else {
          throw new Error("Failed to create session")
        }
      }

      // Send message to the session (with extended timeout for TTS generation)
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 120000) // 2 minute timeout

      const response = await fetch(`${BACKEND_URL}/sessions/${sessionId}/message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage.content,
          patient_id: 'patient101',
          model: model, // Send selected model
          language: language,
          audio_requested: language !== "en-IN"
        }),
        signal: controller.signal,
      })

      clearTimeout(timeoutId)

      if (!response.ok) {
        const errorText = await response.text()
        console.error("Server error:", response.status, errorText)
        throw new Error(`Server error: ${response.status}`)
      }

      const data = await response.json()
      const assistantMessage: Message = {
        role: 'assistant',
        content: data.response,
        audio_content: data.audio_content
      }
      setMessages(prev => [...prev, assistantMessage])

      // Update session title in sidebar if it's the first message
      if (messages.length <= 1) {
        fetchSessions()
      }

    } catch (error) {
      console.error("Chat error:", error)
      const errorMsg = error instanceof Error ? error.message : "Unknown error"
      const errorMessage = { role: 'assistant' as const, content: `I'm sorry, I encountered an error: ${errorMsg}. Please ensure the backend is running.` }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="flex h-full w-full bg-background overflow-hidden">

      <ChatSidebar
        sessions={sessions}
        currentSessionId={currentSessionId}
        onSelectSession={handleSelectSession}
        onNewChat={handleNewChat}
        isOpen={isSidebarOpen}
        onToggle={() => setIsSidebarOpen(!isSidebarOpen)}
      />

      <div className="flex flex-col flex-1 h-full min-w-0 transition-all duration-300">
        {/* Header */}
        <header className="flex items-center justify-between px-4 py-3 border-b shrink-0 h-[60px]">
          <div className='flex items-center gap-3'>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              className="mr-1 text-muted-foreground"
            >
              {isSidebarOpen ? <PanelLeftClose size={20} /> : <PanelLeftOpen size={20} />}
            </Button>

            <div className="relative hidden md:block">
              <Avatar className="h-9 w-9 border">
                <AvatarImage src="/bot-avatar.png" />
                <AvatarFallback className="bg-primary/10 text-primary"><Bot size={18} /></AvatarFallback>
              </Avatar>
              <span className={`absolute bottom-0 right-0 w-2.5 h-2.5 rounded-full border-2 border-background ${model === "gemini-2.5-flash-lite" ? 'bg-blue-500' : model === "medgemma" ? 'bg-purple-500' : 'bg-green-500'}`}></span>
            </div>

            <div>
              <h1 className="text-base font-semibold leading-none flex items-center gap-2">
                Robert
              </h1>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <p className="text-xs text-muted-foreground mt-0.5 cursor-pointer hover:text-foreground transition-colors flex items-center gap-1">
                    {model === "gemini-2.5-flash-lite" ? "Gemini 2.5 Flash" : model === "medgemma" ? "MedGemma 4B (Local)" : "Qwen 0.5B (Local)"} <Settings className="h-3 w-3" />
                  </p>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="start">
                  <DropdownMenuLabel>Model Selection</DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onClick={() => setModel(null)} className="flex justify-between cursor-pointer">
                    Qwen 0.5B (Local) {model === null && "✓"}
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => setModel("medgemma")} className="flex justify-between cursor-pointer">
                    MedGemma 4B (Local) {model === "medgemma" && "✓"}
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => setModel("gemini-2.5-flash-lite")} className="flex justify-between cursor-pointer">
                    Gemini 2.5 Flash {model === "gemini-2.5-flash-lite" && "✓"}
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>

            <div className="ml-1">
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="icon" className="h-8 w-8">
                    <Languages className="h-4 w-4 text-muted-foreground" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="start" className="max-h-[300px] overflow-y-auto">
                  <DropdownMenuLabel>Select Language</DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  {LANGUAGES.map((lang) => (
                    <DropdownMenuItem key={lang.code} onClick={() => setLanguage(lang.code)} className="flex justify-between cursor-pointer">
                      {lang.name} {language === lang.code && "✓"}
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <ModeToggle />
          </div>
        </header>

        {/* Main Chat Area */}
        <div className="flex-1 overflow-hidden relative">
          <ScrollArea className="h-full px-4 md:px-0" ref={scrollAreaRef} >
            <div className="max-w-3xl mx-auto py-6 space-y-6 px-2 md:px-0 pb-10">
              {messages.map((m, index) => (
                <div
                  key={index}
                  className={`flex gap-3 md:gap-4 ${m.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}
                >
                  <Avatar className="h-8 w-8 mt-1 shrink-0">
                    {m.role === 'user' ? (
                      <AvatarFallback className="bg-muted text-muted-foreground"><User size={16} /></AvatarFallback>
                    ) : (
                      <AvatarFallback className="bg-primary/10 text-primary"><Bot size={16} /></AvatarFallback>
                    )}
                  </Avatar>

                  <div className={`group relative max-w-[85%] sm:max-w-[75%] rounded-2xl px-4 py-2.5 text-sm shadow-sm ${m.role === 'user'
                    ? 'bg-primary text-primary-foreground rounded-tr-sm'
                    : 'bg-muted/50 text-foreground border rounded-tl-sm'
                    }`}>
                    <p className="whitespace-pre-wrap leading-relaxed">{m.content}</p>
                    {m.audio_content && (
                      <div className="mt-2 pt-2 border-t border-primary-foreground/20 flex flex-col gap-1">
                        <div className="flex items-center gap-2 text-xs opacity-70 mb-1">
                          <Volume2 size={12} />
                          <span>Audio Response</span>
                        </div>
                        <audio controls src={`data:audio/wav;base64,${m.audio_content}`} className="w-full h-8 max-w-[200px]" />
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {isLoading && (
                <div className="flex gap-4">
                  <Avatar className="h-8 w-8 mt-1 shrink-0">
                    <AvatarFallback className="bg-primary/10 text-primary"><Bot size={16} /></AvatarFallback>
                  </Avatar>
                  <div className="bg-muted/50 border rounded-2xl rounded-tl-sm px-5 py-3 flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                    <span className="text-xs text-muted-foreground">Thinking...</span>
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>
        </div>

        {/* Input Area */}
        <div className="p-3 md:p-4 bg-background border-t">
          <div className="max-w-3xl mx-auto relative">
            <Input
              placeholder="Message Robert..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isLoading}
              className="pr-12 py-6 text-base rounded-full border-muted-foreground/20 focus-visible:ring-offset-0 focus-visible:ring-1 focus-visible:ring-ring bg-muted/20"
            />
            <Button
              onClick={handleSend}
              disabled={isLoading || !input.trim()}
              size="icon"
              className="absolute right-1.5 top-1.5 h-9 w-9 rounded-full"
            >
              <Send className="h-4 w-4" />
              <span className="sr-only">Send</span>
            </Button>
          </div>
          <div className="mt-2 text-[10px] text-center text-muted-foreground hidden sm:block">
            AI can make mistakes. Please verify important medical information.
          </div>
        </div>
      </div>
    </div>
  )
}
