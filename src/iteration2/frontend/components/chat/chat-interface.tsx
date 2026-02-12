'use client'

import { useState, useRef, useEffect } from 'react'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Separator } from "@/components/ui/separator"
import { Send, User, Bot, Loader2, Settings, MoreVertical } from 'lucide-react'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuLabel,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu"
import { ModeToggle } from "@/components/mode-toggle"

interface Message {
  role: 'user' | 'assistant'
  content: string
}

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([
    { role: 'assistant', content: 'Hello! I am Robert. I have access to your medical records. How can I help you understand them today?' }
  ])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [model, setModel] = useState<string | null>(null) // null = default (HF), "gemini-2.5-flash-lite" = Gemini
  const scrollAreaRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (scrollContainer) {
           scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }
    }
  }, [messages])

  const handleSend = async () => {
    if (!input.trim()) return

    const userMessage = { role: 'user' as const, content: input }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    try {
      const response = await fetch('/api/python/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage.content,
          patient_id: 'patient101',
          model: model // Send selected model
        }),
      })

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`)
      }

      const data = await response.json()
      const assistantMessage = { role: 'assistant' as const, content: data.response }
      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
        console.error("Chat error:", error)
        const errorMessage = { role: 'assistant' as const, content: "I'm sorry, I encountered an error connecting to the server. Please ensure the backend is running." }
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
    <div className="flex flex-col h-full w-full bg-background">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b">
        <div className='flex items-center gap-3'>
            <div className="relative">
                <Avatar className="h-10 w-10 border">
                    <AvatarImage src="/bot-avatar.png" />
                    <AvatarFallback className="bg-primary/10 text-primary"><Bot size={20} /></AvatarFallback>
                </Avatar>
                <span className={`absolute bottom-0 right-0 w-3 h-3 rounded-full border-2 border-background ${model ? 'bg-blue-500' : 'bg-green-500'}`}></span>
            </div>
            <div>
                <h1 className="text-lg font-semibold leading-none">Robert</h1>
                <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                        <p className="text-sm text-muted-foreground mt-1 cursor-pointer hover:text-foreground transition-colors flex items-center gap-1">
                            {model ? "Gemini 2.5 Flash" : "Qwen 0.5B (Local)"} <Settings className="h-3 w-3" />
                        </p>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="start">
                        <DropdownMenuLabel>Model Selection</DropdownMenuLabel>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem onClick={() => setModel(null)} className="flex justify-between cursor-pointer">
                            Qwen 0.5B (Local) {model === null && "✓"}
                        </DropdownMenuItem>
                        <DropdownMenuItem onClick={() => setModel("gemini-2.5-flash-lite")} className="flex justify-between cursor-pointer">
                            Gemini 2.5 Flash {model === "gemini-2.5-flash-lite" && "✓"}
                        </DropdownMenuItem>
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
            <div className="max-w-3xl mx-auto py-6 space-y-6">
                {messages.map((m, index) => (
                    <div
                    key={index}
                    className={`flex gap-4 ${m.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}
                    >
                        <Avatar className="h-8 w-8 mt-1 shrink-0">
                            {m.role === 'user' ? (
                                <AvatarFallback className="bg-muted text-muted-foreground"><User size={16} /></AvatarFallback>
                            ) : (
                                <AvatarFallback className="bg-primary/10 text-primary"><Bot size={16} /></AvatarFallback>
                            )}
                        </Avatar>
                        
                        <div className={`group relative max-w-[85%] rounded-2xl px-5 py-3 text-sm shadow-sm ${
                            m.role === 'user'
                            ? 'bg-primary text-primary-foreground rounded-tr-sm'
                            : 'bg-muted/50 text-foreground border rounded-tl-sm'
                        }`}>
                            <p className="whitespace-pre-wrap leading-relaxed">{m.content}</p>
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
                 <div className="h-4" /> {/* Spacer */}
            </div>
        </ScrollArea>
      </div>

      {/* Input Area */}
      <div className="p-4 bg-background border-t">
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
         <div className="mt-2 text-[10px] text-center text-muted-foreground">
            AI can make mistakes. Please verify important medical information.
        </div>
      </div>
    </div>
  )
}
